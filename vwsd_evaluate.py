import torch
import torch.nn.functional as F
import numpy as np
from load_dataset import get_vwsd_dataloader
from utils import cal_metrics,convert_models_to_fp32
from tqdm import tqdm
import time

@torch.no_grad()
def eval_module(model,sentence,img_names,candidate_images):
    text_encoder = model.text_encoder

    sentence_feats,_ = text_encoder(model.tokenize(sentence, truncate=True).to(model.device))
    sentence_feats = F.normalize(sentence_feats, dim=-1)

    pred_imgs = []
    sort_ten = []

    # 得到每个歧义短语对应的10张候选图片的特征
    candidate_img_feats,image_tokens_embedding = model.vision_encoder(candidate_images.to(model.device))
    candidate_img_feats = F.normalize(candidate_img_feats, dim=-1)

    image_tokens_embedding = model.image_seq_transfer(image_tokens_embedding)
    image_tokens_embedding = F.normalize(image_tokens_embedding, dim=-1)
    image2glo_embedding, image2glo_tokens_embedding = model.img2gloss_former(image_tokens_embedding)
    image2glo_multi_embedding = model.multi_fusion(image_tokens_embedding, image2glo_tokens_embedding,mode='image_guided')
    image2glo_multi_embedding = F.normalize(image2glo_multi_embedding, dim=-1)

    image_conprehensive_embedding = candidate_img_feats + image2glo_multi_embedding


    # 计算logits
    sim_logits = sentence_feats @ image_conprehensive_embedding.T
    # sim_logits = sentence_feats @ candidate_img_feats.T
    # sim_logits = sentence_feats @ candidate_img_feats.T  # 计算每个歧义短语和10张候选图片的相似度矩阵
    final_logits = sim_logits

    # 确定预测图片
    logits_numpy = final_logits.softmax(1).detach().cpu().numpy()
    max_index = np.argmax(logits_numpy)
    pred = img_names[max_index]
    pred_imgs.append(pred)

    # 记录结果
    _, idx_topk = torch.topk(final_logits, k=10, dim=-1)
    result = []
    for j in idx_topk[0]:
        j = int(j)
        result.append(img_names[j])
    sort_ten.append(result)

    return pred_imgs, sort_ten

@torch.no_grad()
def evaluation_fn(epoch, args, data, model):
    model.eval()
    convert_models_to_fp32(model)
    eval_dataloader = get_vwsd_dataloader(args, data)
    loop = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), ncols=150)

    count = 0
    acc_sum = 0
    GOLD = []
    best_imgs = []
    SORT10 = []

    idx = 0
    for _, data_list in loop:
        for data in data_list:
            sentence = data['sentence']
            img_names = data['candidate_imgs']
            candidate_images = data['candidate_images_vec']
            gold_img = data['gold_img']
            GOLD.append(gold_img)

            # start evaluating
            with torch.no_grad():
                pred_imgs, sort_ten = eval_module(model,sentence,img_names,candidate_images)

            ### record
            best_imgs += pred_imgs
            SORT10.append(sort_ten)

            # calculate the accuracy
            for i, pred in enumerate(pred_imgs):
                if pred == gold_img:
                    acc_sum += 1

            count += 1
            now_acc,now_mrr = cal_metrics(SORT10, GOLD, count)

            # update the loop message
            # loop.set_description(f'{mode} Epoch [{epoch + 1}/{args.epochs}] Evaluating [{idx + 1}/{len(loop)}]')
            idx += 1
            loop.set_description(f'Visual WSD# Epoch [{epoch + 1}/{args.epochs}] Evaluating [{idx + 1}/{len(loop)}] Acc:{now_acc:.4f} Mrr:{now_mrr:.4f}')

    # calculate metrics
    acc, mrr = cal_metrics(SORT10, GOLD, count)
    # time.sleep(0.2)
    # print('Evaluating Accuracy = ', acc, ' MRR = ', mrr)

    status = {
        'visual_test_acc': acc,
        'visual_test_mrr': mrr
    }

    return status