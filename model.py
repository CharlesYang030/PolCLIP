import torch
from backbone.CLIP import clip
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

class mymodel(nn.Module):
    def __init__(self, args):
        super(mymodel, self).__init__()
        self.args = args
        self.device = args.device

        model, _ = clip.load(args.model_name,device=args.device, jit=False)
        self.tokenize = clip.tokenize

        # create normal encoders
        self.vision_encoder = visual_encoder_withparas(model)
        self.text_encoder = text_encoder_withparas(model)

        self.former = FormerModel(layers=2)
        self.gloss2img_projector = nn.Linear(768, 1024, bias=True)
        self.gloss2img_visual_encoder = gloss2img_visual_encoder(self.vision_encoder,last_layers_num=2)
        self.img2gloss_projector = nn.Linear(768, 768, bias=True)
        self.img2gloss_text_encoder = img2gloss_text_encoder(self.text_encoder, last_layers_num=2)
        self.seq_transfer = nn.Linear(257, 77, bias=True)

        ## SIG
        self.gloss2img_former = nn.Sequential(
            self.former,
            self.gloss2img_projector,
            self.gloss2img_visual_encoder
        )

        ## ISG
        self.img2gloss_former = nn.Sequential(
            self.former,
            self.img2gloss_projector,
            self.img2gloss_text_encoder
        )

        self.multi_fusion = Multi_Fusion(num_layers=4)

        self.momentum = 0.995
        # create momentum encoders
        self.vision_encoder_m = visual_encoder_withparas(model)
        self.text_encoder_m = text_encoder_withparas(model)

        self.model_pairs = [[self.vision_encoder, self.vision_encoder_m],
                            [self.text_encoder, self.text_encoder_m],
                            ]
        self.copy_params()

    def get_gloss2img_labels(self,data):
        labels = torch.zeros([len(data['gloss2image_labels']), len(data['total_candidate_image'])]).float().to(self.device)
        for i,tup in enumerate(data['gloss2image_labels']):
            labels[i,tup[0]:tup[1]] = 1
        return labels

    def get_sent2glossmul_labels(self,data):
        labels = torch.zeros([len(data['sentence2gloss_labels']), len(data['total_candidate_gloss'])]).float().to(self.device)
        for i,(gold,rang) in enumerate(zip(data['sentence2gloss_labels'],data['candidate_gloss_labels'])):
            labels[i,rang[0]:rang[1]][gold] = 1
        return labels

    def get_image2glo_labels(self,data):
        labels = torch.zeros([len(data['total_candidate_image']), len(data['total_candidate_gloss'])]).float().to(self.device)
        for i,gold in enumerate(data['image2gloss_labels']):
            labels[i,gold] = 1
        return labels

    def get_sent2imagemul_labels(self,data):
        labels = torch.zeros([len(data['sentence2image_labels']), len(data['total_candidate_image'])]).float().to(self.device)
        for i,rang in enumerate(data['sentence2image_labels']):
            labels[i,rang[0]:rang[1]] = 1
        return labels

    def image_seq_transfer(self,x):
        x = x.permute(0,2,1)
        x = self.seq_transfer(x)
        x = x.permute(0,2,1)
        return x

    def get_captions(self,data):
        captions = []
        for i,w in enumerate(data['word']):
            candidate_gloss = data['candidate_gloss'][i]
            for glo in candidate_gloss:
                caption_str = f'A photo of "{w}", {glo.lower()}.'
                captions.append(caption_str)
        return captions

    def forward(self,data):
        sentence_embedding,_ = self.text_encoder(data['sentence_tokens'],mask=data['sentence_mask'])
        sentence_embedding = F.normalize(sentence_embedding, dim=-1)

        gloss_embedding,gloss_tokens_embedding = self.text_encoder(data['gloss_tokens']) #gloss_embed=[gloss_batch,width],gloss_tokens_embedding=[gloss_batch,sequence,width]
        gloss2img_embedding,gloss2img_tokens_embedding = self.gloss2img_former(gloss_tokens_embedding) #gloss2img_embedding=[gloss_batch,width]  gloss2img_tokens_embedding=[gloss_batch,seq,width]
        gloss2img_tokens_embedding = F.normalize(gloss2img_tokens_embedding,dim=-1)
        gloss2img_multi_embedding = self.multi_fusion(gloss_tokens_embedding,gloss2img_tokens_embedding,mode='gloss_guided') #gloss2img_multi_embedding = [gloss_batch,width]

        gloss_embedding = F.normalize(gloss_embedding,dim=-1)
        gloss2img_embedding = F.normalize(gloss2img_embedding, dim=-1)
        gloss2img_multi_embedding = F.normalize(gloss2img_multi_embedding, dim=-1)
        with torch.no_grad():
            self._momentum_update()
            image_embedding_m = F.normalize(self.vision_encoder_m(data['images_tokens'])[0], dim=-1)
            gloss_embedding_m = F.normalize(self.text_encoder_m(data['gloss_tokens'])[0], dim=-1)

        # ### LOSS 1: SIG √
        gloss2img_similarity = gloss2img_embedding @ image_embedding_m.T
        gloss2img_labels = self.get_gloss2img_labels(data)
        loss1 = -torch.sum(F.log_softmax(gloss2img_similarity, dim=1) * gloss2img_labels, dim=1).mean()

        ### LOSS 2: W2S √
        sent2gloss_similarity = sentence_embedding @ gloss2img_multi_embedding.T
        sent2glossmul_labels = self.get_sent2glossmul_labels(data)
        loss2 = -torch.sum(F.log_softmax(sent2gloss_similarity, dim=1) * sent2glossmul_labels, dim=1).mean()

        #####################################################
        #####################################################
        image_embedding, image_tokens_embedding = self.vision_encoder(data['images_tokens'])
        image_tokens_embedding = self.image_seq_transfer(image_tokens_embedding)
        image_tokens_embedding = F.normalize(image_tokens_embedding, dim=-1)
        image2glo_embedding, image2glo_tokens_embedding = self.img2gloss_former(image_tokens_embedding)
        image2glo_multi_embedding = self.multi_fusion(image_tokens_embedding, image2glo_tokens_embedding,mode='image_guided')

        image_embedding = F.normalize(image_embedding, dim=-1)
        image2glo_embedding = F.normalize(image2glo_embedding, dim=-1)
        image2glo_multi_embedding = F.normalize(image2glo_multi_embedding, dim=-1)

        ### LOSS 3: ISG √
        image2glo_similarity = image2glo_embedding @ gloss_embedding_m.T
        image2glo_labels = self.get_image2glo_labels(data)
        loss3 = -torch.sum(F.log_softmax(image2glo_similarity, dim=1) * image2glo_labels, dim=1).mean()

        ### LOOS 4: W2I √
        captions = self.get_captions(data)
        captions_embedding, _ = self.text_encoder(self.tokenize(captions, truncate=True).to(self.device))
        captions_embedding = F.normalize(captions_embedding,dim=-1)
        caption2img_similarity = captions_embedding @ image2glo_multi_embedding.T
        caption2img_labels = gloss2img_labels
        loss4 = -torch.sum(F.log_softmax(caption2img_similarity, dim=1) * caption2img_labels, dim=1).mean()

        loss = loss1 + loss2 + loss3 + loss4

        ### evaluate predictions
        bingo_num = 0
        instance_num = len(data['word'])
        for ins in range(instance_num):
            sent2glossmul_logits = sent2gloss_similarity[ins].unsqueeze(0).detach().cpu().numpy()
            sent2glossmul_logits = sent2glossmul_logits[:,data['candidate_gloss_labels'][ins][0]:data['candidate_gloss_labels'][ins][1]]
            max_index = np.argmax(sent2glossmul_logits)
            if max_index == data['sentence2gloss_labels'][ins]:
                bingo_num += 1

        caption_bingo_num = 0
        caption_instance_num = len(data['total_candidate_gloss'])
        caption2image_target = data['gloss2image_labels']
        for ins in range(caption_instance_num):
            caption2img_logits = caption2img_similarity[ins].unsqueeze(0).detach().cpu().numpy()
            max_index = np.argmax(caption2img_logits)
            left = caption2image_target[ins][0]
            right = caption2image_target[ins][1] - 1
            if left<= max_index <= right:
                caption_bingo_num += 1

        return loss,loss1,loss2,loss3,loss4,bingo_num,instance_num,caption_bingo_num,caption_instance_num


    @torch.no_grad()
    def copy_params(self):
        for idx, model_pair in enumerate(self.model_pairs):
            if idx == 0:  # idx == 0 表示VisualTransformer
                for param, param_m in zip(model_pair[0].parameters(),model_pair[1].parameters()):
                    param_m.data.copy_(param.data)  # initialize
                    param_m.requires_grad = False  # not update by gradient
            else:
                for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                    param_m.data.copy_(param.data)  # initialize
                    param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for idx, model_pair in enumerate(self.model_pairs):
            if idx == 0:  # idx == 0 表示VisualTransformer
                for param, param_m in zip(model_pair[0].transformer.parameters(),
                                          model_pair[1].transformer.parameters()):
                    param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
            else:
                for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                    param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
            if idx != 0:
                for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                    param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

class visual_encoder_withparas(nn.Module):
    def __init__(self,clip_model):
        super(visual_encoder_withparas, self).__init__()
        self.transformer = clip_model.visual.transformer.float()
        self.dtype = torch.float16

        width = 1024
        input_resolution = 224
        out_dim = 768
        patch_size = 14
        self.input_resolution = input_resolution
        self.output_dim = out_dim
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        tokens_x = x
        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
            tokens_x = tokens_x @ self.proj
        return x, tokens_x

        # img_feat = self.visual_encoder(image.type(self.dtype))
        # return img_feat.float()

class gloss2img_visual_encoder(nn.Module):
    def __init__(self,complete_visual_encoder,last_layers_num=4):
        super(gloss2img_visual_encoder, self).__init__()
        self.dtype = torch.float32

        width = 1024
        # input_resolution = 224
        out_dim = 768
        # patch_size = 14
        # self.input_resolution = input_resolution
        self.output_dim = out_dim

        scale = width ** -0.5
        transformer = complete_visual_encoder.transformer.float()
        self.transformer_last_layers = transformer.resblocks[-last_layers_num:]

        self.ln_post = complete_visual_encoder.ln_post
        self.proj = complete_visual_encoder.proj

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_last_layers(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        temp_x = self.ln_post(torch.mean(x,dim=1))
        #
        if self.proj is not None:
            x = x @ self.proj
            temp_x = temp_x @ self.proj
        return temp_x, x

class text_encoder_withparas(nn.Module):
    def __init__(self,clip_model):
        super(text_encoder_withparas, self).__init__()
        self.token_embedding = clip_model.token_embedding.float()
        self.positional_embedding = clip_model.positional_embedding.float()
        self.transformer = clip_model.transformer.float()
        self.ln_final = clip_model.ln_final.float()
        self.text_projection = clip_model.text_projection.float()
        self.dtype = torch.float32

    def forward(self,text,mask=None):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        token_x = x.float()

        # x.shape = [batch_size, n_ctx, transformer.width]
        if not mask:
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        else:
            # take features from the mask index (mask index is the target word in each sequence)
            temp_x = []
            for i in range(len(mask)):
                t = x[i][mask[i][0]:mask[i][1],:]
                temp_x.append(torch.mean(t,dim=0).unsqueeze(0))
            x = torch.vstack(temp_x) @ self.text_projection

        return x.float(),token_x

class img2gloss_text_encoder(nn.Module):
    def __init__(self,complete_text_encoder,last_layers_num=4):
        super(img2gloss_text_encoder, self).__init__()
        transformer = complete_text_encoder.transformer.float()
        self.transformer_last_layers = transformer.resblocks[-last_layers_num:]
        self.ln_final = complete_text_encoder.ln_final.float()
        self.text_projection = complete_text_encoder.text_projection.float()
        self.dtype = torch.float32

    def forward(self,text):
        x = text.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_last_layers(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        token_x = x.float()

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = torch.mean(x,dim=1) @ self.text_projection
        return x,token_x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


## former module
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class FormerModel(nn.Module):
    def __init__(self, width=768, layers=2, heads=8, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2)
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)
        return x

###  cross-attention module
class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def forward(self, query, key, value):
        # query, key, value 的形状：[seq_len, batch_size, d_model]
        query_att = query + self.multihead_attn(self.ln_1(query), self.ln_1(key), self.ln_1(value))[0]
        query_att = query_att + self.mlp(self.ln_2(query_att))
        return query_att

    # def forward(self, query, key, value):
    #     # query, key, value 的形状：[seq_len, batch_size, d_model]
    #     attn_output, _ = self.multihead_attn(query, key, value)
    #     return attn_output

class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class Multi_Fusion(nn.Module):
    def __init__(self, d_model=768, num_heads=8, dim_feedforward=2048, num_layers=4,attn_mask: torch.Tensor = None):
        super(Multi_Fusion, self).__init__()
        self.layers = nn.ModuleList([CrossAttentionLayer(d_model, num_heads) for _ in range(num_layers)])
        self.feed_forward = FeedForward(d_model, dim_feedforward)
        self.ln_post = LayerNorm(d_model)

    def forward(self, text, image,mode=None):
        # text, image 的形状：[seq_len, batch_size, d_model]
        text = text.permute(1, 0, 2)
        image = image.permute(1, 0, 2)

        if mode == 'gloss_guided':
            for layer in self.layers:
                text = layer(text, image, image)  # text作为query, image作为key和value
            output = self.feed_forward(text)
        elif mode == 'image_guided':
            for layer in self.layers:
                image = layer(image, text, text)  # text作为query, image作为key和value
            output = self.feed_forward(image)
        output = output.permute(1, 0, 2)
        output = self.ln_post(output[:,0,:])
        return output

