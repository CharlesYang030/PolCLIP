import json
import torch
import argparse
from utils import seed_everything

import os.path
import os
import sys
import wandb
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from pytorch_transformers import AdamW,WarmupLinearSchedule
from random import sample
from model import mymodel
from utils import frozen_model,calculate_modelpara,cosine_lr_schedule,convert_models_to_fp32
from torch.cuda.amp import autocast, GradScaler
from load_dataset import get_data,get_dataloader
from vwsd_evaluate import evaluation_fn
from evaluate import _evaluate


import warnings
warnings.filterwarnings("ignore")

torch.distributed.init_process_group(backend="nccl")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def set_config():
    parser = argparse.ArgumentParser(description="Multimodal-WSD")
    parser.add_argument('--wandb_name', type = str, default="PolCLIP for Multimodal-WSD")
    parser.add_argument('--project_name',type=str,required=False, default='PolCLIP for Multimodal-WSD',help='a name of this project')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--train_data', type=str, default='all', choices=['ie_semcor', 'all'],help='all means ie_semcor + vwsdkb')
    parser.add_argument('--generate_tokens', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gloss_bsz', type=int, default=50, choices=[20, 50])
    parser.add_argument('--image_bsz', type=int, default=250, choices=[100, 250])
    parser.add_argument('--epochs', type=int, required=False, default=20)
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help="learning rate")
    parser.add_argument('--min_lr', type=float, required=False, default=0.0, help="minimum learning rate")
    parser.add_argument('--weight_decay', type=float, required=False, default=0.05, help="weight_decay")
    parser.add_argument('--patience', type=int, required=False, default=6, help="patience for rlp")
    
    parser.add_argument('--text_encoder_open_layer', type=int, default=12)
    parser.add_argument('--visual_encoder_open_layer', type=int, default=0)

    parser.add_argument('--semcor_training_data_path',  type=str, default=r'/home/data/semcor_training_tokens.50.250.json')
    parser.add_argument('--vwsd_training_data_path',  type=str, default=r'/home/data/vwsd_training_tokens.50.250.json')
    parser.add_argument('--twsd_eval_data_path', type=str,default=r'/home/data/all_twsd_test_data.json')
    parser.add_argument('--vwsd_test_path', type=str,default=r'/home/data/VWSD_testdata.json')
    parser.add_argument('--one_time_read_images', action='store_true', default=False, help='whether read images at one time')
    parser.add_argument('--num_workers', type=int, default=16)

    parser.add_argument('--model_name', type=str, default='ViT-L/14',choices=['ViT-L/14'])

    parser.add_argument('--use_checkpoint', action='store_true', default=False, help="use pretrained weights or not")
    parser.add_argument('--evaluate', action='store_true', default=False, help="evaluate or not")



    args = parser.parse_args()
    return args


def get_img_vec(image_path):
    import pickle
    with open(image_path, 'rb') as f:
        image_vec = pickle.load(f)
        key = list(image_vec.keys())[0]
        image_vec = image_vec[key].unsqueeze(0)
    return image_vec

def get_image_tokens(candidate_image):
    vecs = []
    for path in candidate_image:
        if 'pkl' in path:
            image_vec = get_img_vec(path)
        else:
            image_vec = torch.load(path).unsqueeze(0)
        vecs.append(image_vec)
    vecs = torch.vstack(vecs)
    return vecs

def _train(args, epoch, train_dataloader,total_num, model, optimizer, scaler):
    model.train()
    convert_models_to_fp32(model)
    LOSS = []
    L1,L2,L3,L4 = [],[],[],[]
    LR = []
    Bingo_n = 0
    Instance_n = 0
    vwsd_Bingo_n = 0
    vwsd_Instance_n = 0
    total_num = len(train_dataloader)
    # train_dataloader = get_dataloader(args, train_data, mode='train')
    loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=180)

    idx = 0
    for _, data_list in loop:
        for data in data_list:
            # data = transfer_dict(data)
            data['sentence_tokens'] = torch.tensor(data['sentence_tokens'], dtype=torch.int32)
            data['gloss_tokens'] = torch.tensor(data['gloss_tokens'], dtype=torch.int32)
            data['images_tokens'] = get_image_tokens(data['total_candidate_image'])
            # try:
            #     data['images_tokens'] = torch.vstack([torch.load(x).unsqueeze(0) for x in data['total_candidate_image']])
            # except:
            #     data['images_tokens'] = torch.vstack([get_img_vec(x) for x in data['total_candidate_image']])

            data['sentence_tokens'] = data['sentence_tokens'].to(args.device)
            data['gloss_tokens'] = data['gloss_tokens'].to(args.device)
            data['images_tokens'] = data['images_tokens'].to(args.device)

            # start training
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            with autocast():
                loss,loss1,loss2,loss3,loss4,bingo_num,instance_num,caption_bingo_num,caption_instance_num = model(data)

            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            convert_models_to_fp32(model)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.cuda.empty_cache()

            # record
            loss1, loss2, loss3,loss4 = loss1.detach().cpu().numpy(),loss2.detach().cpu().numpy(),loss3.detach().cpu().numpy(),loss4.detach().cpu().numpy()
            loss1, loss2, loss3,loss4 = float(loss1), float(loss2), float(loss3), float(loss4)
            L1.append(loss1)
            L2.append(loss2)
            L3.append(loss3)
            L4.append(loss4)
            loss_np = loss.detach().cpu().numpy()
            LOSS.append(float(loss_np))
            LR.append(optimizer.param_groups[0]["lr"])
            Bingo_n += bingo_num
            Instance_n += instance_num
            textual_acc = Bingo_n / Instance_n
            vwsd_Bingo_n += caption_bingo_num
            vwsd_Instance_n += caption_instance_num
            visual_acc = vwsd_Bingo_n / vwsd_Instance_n

            # update the loop message
            idx += 1
            loop.set_description(
                f'Epoch [{epoch + 1}/{args.epochs}] Training [{idx}/{total_num}] All Loss: {round(np.mean(LOSS), 6)} LR: {round(float(LR[-1]),6)} ||| '
                f'Now: l1:{round(np.mean(L1),4)} l2:{round(np.mean(L2),4)} l3:{round(np.mean(L3),4)} l4:{round(np.mean(L4),4)} Acc1:{round(textual_acc,4)} Acc2:{round(visual_acc,4)}')

    status = {
        'loss': np.mean(LOSS),
        'L1': np.mean(L1),
        'L2': np.mean(L2),
        'L3': np.mean(L3),
        'L4': np.mean(L4),
        'lr': LR[-1],
        'textual_training_acc': round(Bingo_n / Instance_n,6),
        'visual_training_acc': round(vwsd_Bingo_n / vwsd_Instance_n,6)
    }

    return status


def train_model(args):
    if torch.distributed.get_rank() == 0:
        wandb.init(project='PolCLIP for Multimodal-WSD', name = args.wandb_name, config = args)
    
    print('Loading model...')
    sys.stdout.flush()

    print('Loading data + preprocessing...')
    sys.stdout.flush()
    print('train_data_size: ', args.train_data)
    if args.train_data == 'ie_semcor':
        semcor_train_data = json.load(open(args.semcor_training_data_path,'r',encoding='utf-8'))
        train_data = semcor_train_data
    elif args.train_data == 'all':
        print(args.semcor_training_data_path)
        print(args.vwsd_training_data_path)
        semcor_train_data = json.load(open(args.semcor_training_data_path,'r',encoding='utf-8'))
        vwsd_train_data = json.load(open(args.vwsd_training_data_path,'r',encoding='utf-8'))
        train_data = semcor_train_data + vwsd_train_data

    total_num = len(train_data)
    print('Total data: ',len(train_data))
    eval_data = json.load(open(args.twsd_eval_data_path,'r',encoding='utf-8'))
    vwsd_test_data = json.load(open(args.vwsd_test_path, 'r', encoding='utf-8'))
    
    ### split semeval test data
    semeval_class=['02','03','07','10','13','15']
    class_split = {}
    for cls in semeval_class:
        eval_data_temp = []
        for eval_d in eval_data:
            if eval_d['class'] == cls:
                eval_data_temp.append(eval_d)
        class_split[cls] = get_dataloader(args, eval_data_temp, mode='eval')
        
    pos_class = ['NOUN', 'VERB', 'ADJ', 'ADV']
    pos_split = {}
    for pos in pos_class:
        eval_data_temp = []
        for eval_d in eval_data:
            if eval_d['pos'] == pos:
                eval_data_temp.append(eval_d)
        pos_split[pos] = get_dataloader(args, eval_data_temp, mode='eval')
    ###
            

    train_dataloader = get_dataloader(args, train_data,mode='train')
    eval_dataloader = get_dataloader(args, eval_data, mode='eval')

    # 3. set ranks
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    args.device = torch.device("cuda", local_rank)

    model = mymodel(args)
    if args.use_checkpoint:
        model.load_state_dict(torch.load('checkpoint.pt'))
        print('>>>>>>Using the model checkpoint :\n')
    frozen_model(model,args,mode='partial')
    if torch.distributed.get_rank() == 0:
        calculate_modelpara(model)

    model.to(args.device)
    if torch.distributed.get_rank() == 0:
        wandb.watch(model, log='all')
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    weight_decay = args.weight_decay  # this could be a parameter
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if 'vision_encoder' in n],'lr':args.lr*0.5},
        {'params': [p for n, p in model.named_parameters() if 'text_encoder' in n],'lr':args.lr},
        {'params': [p for n, p in model.named_parameters() if 'vision_encoder' not in n and 'text_encoder' not in n],'lr':args.lr*0.5}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=args.weight_decay)
    # optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()

    '''
    TRAIN MODEL ON SEMCOR DATA
    '''
#     if args.use_checkpoint and torch.distributed.get_rank() == 0:
#             print('### Evaluation before training')
#             # eval_status = _evaluate(args, -1, eval_dataloader, model.module)  # pre evaluate
#             # vwsd_status = evaluation_fn(-1, args, vwsd_test_data, model.module)  # VWSD eval
            
#             semeval_status = {}
#             for cls,dataloader in class_split.items():
#                 print('### Evaluating ' + cls)
#                 eval_status = _evaluate(args, -1, dataloader, model.module)
#                 semeval_status[cls] = eval_status['textual_test_acc']
#             for pos,dataloader in pos_split.items():
#                 print('### Evaluating ' + pos)
#                 eval_status = _evaluate(args, -1, dataloader, model.module)
#                 semeval_status[pos] = eval_status['textual_test_acc']
            
#             print(semeval_status)
#             exit()

#     print('### Evaluation before training')
#     eval_status = _evaluate(args, -1, eval_dataloader, model.module)  # pre evaluate
#     vwsd_status = evaluation_fn(-1, args, vwsd_test_data, model.module)  # VWSD eval

#     semeval_status = {}
#     for cls,dataloader in class_split.items():
#         print('### Evaluating ' + cls)
#         eval_status = _evaluate(args, -1, dataloader, model.module)
#         semeval_status[cls] = eval_status['textual_test_acc']
#     for pos,dataloader in pos_split.items():
#         print('### Evaluating ' + pos)
#         eval_status = _evaluate(args, -1, dataloader, model.module)
#         semeval_status[pos] = eval_status['textual_test_acc']

#     print(semeval_status)
#     exit()

    print('Training unified disambiguation model...')
    for epoch in range(args.epochs):
        cosine_lr_schedule(optimizer, epoch+7, args.patience, args.lr, args.min_lr)
        print('======' * 10, f'Epoch {str(epoch + 1)}', '======' * 10)
        train_status = _train(args, epoch, train_dataloader,total_num, model, optimizer, scaler)
        eval_status = _evaluate(args, epoch, eval_dataloader, model.module)     # Textual WSD eval
        vwsd_status = evaluation_fn(epoch, args, vwsd_test_data, model.module)  # Visual WSD eval
        if torch.distributed.get_rank() == 0:
            torch.save(model.module.state_dict(), f'checkpoint.pt')
            
            log_dict = {}
            for k,v in train_status.items():
               log_dict[k] = v
            for k,v in eval_status.items():
               log_dict[k] = v 
            for k,v in vwsd_status.items():
               log_dict[k] = v 
            
            print('rank 0:')
            for cls,dataloader in class_split.items():
                print('### Evaluating ' + cls)
                eval_status = _evaluate(args, -1, dataloader, model.module)
                log_dict[cls] = eval_status['textual_test_acc']
            for pos,dataloader in pos_split.items():
                print('### Evaluating ' + pos)
                eval_status = _evaluate(args, -1, dataloader, model.module)
                log_dict[pos] = eval_status['textual_test_acc']
            
            wandb.log(log_dict)
            print(log_dict)
            
            

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("Need available GPU(s) to run this model...")
        quit()

    # parse args
    args = set_config()
    seed_everything(args.seed)
    
    # evaluate model saved at checkpoint or...
    if args.evaluate:
        # evaluate_model(args)
        print()
    # train model
    else:
        train_model(args)