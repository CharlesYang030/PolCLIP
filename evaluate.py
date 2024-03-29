from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from load_dataset import get_dataloader
from utils import convert_models_to_fp32

@torch.no_grad()
def _evaluate(args,epoch,eval_dataloader,model):
    model.eval()
    convert_models_to_fp32(model)
    LOSS = []
    Bingo_n = 0
    Instance_n = 0
    # eval_dataloader = get_dataloader(args, eval_data, mode='eval')
    loop = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), ncols=180, colour='blue')
    
    idx = 0
    for _, data_list in loop:
        for data in data_list:
            data['sentence_tokens'] = torch.tensor(data['sentence_tokens'][0], dtype=torch.int32)
            data['sentence_mask'] = data['sentence_mask'][0]
            data['gloss_tokens'] = torch.tensor(data['gloss_tokens'][0], dtype=torch.int32)
            data['sentence_tokens'] = data['sentence_tokens'].to(args.device)
            data['gloss_tokens'] = data['gloss_tokens'].to(args.device)

            sentence_embedding, _ = model.text_encoder(data['sentence_tokens'], mask=data['sentence_mask'])
            sentence_embedding = F.normalize(sentence_embedding, dim=-1)

            gloss_embedding, gloss_tokens_embedding = model.text_encoder(data['gloss_tokens'])  # gloss_embed=[gloss_batch,width],gloss_tokens_embedding=[gloss_batch,sequence,width]
            gloss2img_embedding, gloss2img_tokens_embedding = model.gloss2img_former(gloss_tokens_embedding)  # gloss2img_embedding=[gloss_batch,width]  gloss2img_tokens_embedding=[gloss_batch,seq,width]
            gloss2img_tokens_embedding = F.normalize(gloss2img_tokens_embedding, dim=-1)
            gloss2img_multi_embedding = model.multi_fusion(gloss_tokens_embedding, gloss2img_tokens_embedding,mode='gloss_guided')  # gloss2img_multi_embedding = [gloss_batch,width]

            gloss_embedding = F.normalize(gloss_embedding,dim=-1)
            gloss2img_multi_embedding = F.normalize(gloss2img_multi_embedding, dim=-1)

            gloss_comprehensive_embedding = gloss_embedding + gloss2img_multi_embedding

            sent2glossmul_similarity = sentence_embedding @ gloss_comprehensive_embedding.T
            # sent2glossmul_similarity = sentence_embedding @ gloss_embedding.T
            # sent2glossmul_similarity = sentence_embedding @ gloss2img_multi_embedding.T   # sent 2 gloss mul   ACC: 0.364 Eval Loss: 1.37
            # sent2glossmul_similarity = sentence_embedding @ F.normalize(gloss_embedding, dim=-1).T  # sent 2 pure gloss   ACC: 0.5028 Eval Loss: 1.361765
            sent2glossmul_labels = model.get_sent2glossmul_labels(data)
            loss = -torch.sum(F.log_softmax(sent2glossmul_similarity, dim=1) * sent2glossmul_labels,dim=1).mean()  # 与batch中的所有gloss消歧，增加了消歧难度

            ### evaluate predictions
            sent2glossmul_logits = sent2glossmul_similarity.detach().cpu().numpy()
            sent2glossmul_logits = sent2glossmul_logits[:,data['candidate_gloss_labels'][0][0]:data['candidate_gloss_labels'][0][1]]
            max_index = np.argmax(sent2glossmul_logits)
            if data['candidate_labels'][max_index] in data['gold_labels']:
                Bingo_n += 1

            # record
            loss = loss.detach().cpu().numpy()
            LOSS.append(float(loss))
            Instance_n += 1
            acc = Bingo_n / Instance_n

            # update the loop message
            idx += 1
            loop.set_description(
                f'Textual WSD# Epoch [{epoch + 1}/{args.epochs}] Evaluating [{idx + 1}/{len(loop)}] Eval Loss: {round(np.mean(LOSS), 6)} ||| ACC:{round(acc, 4)}')

    status = {
        'textual_test_loss': np.mean(LOSS),
        'textual_test_acc': round(Bingo_n / Instance_n,6)
    }

    return status

