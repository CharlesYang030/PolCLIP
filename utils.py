import random
import os
from datetime import datetime
import numpy as np
import torch
import math


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def calculate_modelpara(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})

# 冻住模型指定的层数
def frozen_model(model,args,mode):
    names = []
    values = []
    for name, value in model.named_parameters():
        names.append(name)
        values.append(value)
        value.requires_grad = False
    if mode == 'partial':
        ### 冻结clip中的某些层
        grad_frozen = []
        for i in range(0,24-args.visual_encoder_open_layer):
            grad_frozen.append('vision_encoder.transformer.resblocks.'+str(i)) # vision transformer frezon layers
        for i in range(0,12-args.text_encoder_open_layer):
            grad_frozen.append('text_encoder.transformer.resblocks.' + str(i))  # vision transformer frezon layers
        grad_frozen += [
            'vision_encoder.conv1','vision_encoder.class_embedding','vision_encoder.positional_embedding','vision_encoder.ln_pre',
            'text_encoder.positional_embedding','text_encoder.token_embedding'
        ]
        for name, value in model.named_parameters():
            frezon_flag = False
            for g in grad_frozen:
                if g in name:
                    frezon_flag = True
                    break
                    
            import re
            matches = re.findall(r"resblocks\.\d+", name)
            if matches != []:
                layer = int(matches[0].split('.')[-1])
                if 'vision_encoder' in name and layer >= 24-args.visual_encoder_open_layer:
                    frezon_flag = False
                elif 'text_encoder' in name and layer >= 12-args.text_encoder_open_layer:
                    frezon_flag = False
                    
            if frezon_flag:
                value.requires_grad = False
            else:
                value.requires_grad = True
    elif mode == 'none':
        for name, value in model.named_parameters():
            value.requires_grad = True


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def convert_models_to_fp32(model):
    for p in model.parameters():
        if not p.data == None:
            p.data = p.data.float()
        if not p.grad == None:
            p.grad.data = p.grad.data.float()

def cal_metrics(SORT10,GOLD,total):
    # gold_imgs = []
    # for golds in GOLD:
    #     for g in golds:
    #         gold_imgs.append(g)

    gold_imgs = GOLD
    acc = 0
    mrr = 0
    count = 0
    for prediction in SORT10:
        batch = len(prediction)
        for i in range(batch):
            gold = gold_imgs[count + i]
            pred_best = prediction[i][0]
            # acc
            if gold == pred_best:
                acc += 1
            # mrr
            for j in range(len(prediction[i])):
                if gold == prediction[i][j]:
                    # 注意j的取值从0开始
                    mrr += 1/(j+1)
                    break
        count += batch
    acc = acc / total * 100.
    mrr = mrr / total * 100.
    return acc,mrr