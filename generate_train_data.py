import os.path
from random import sample
from model import mymodel
import argparse
import torch
import json
from tqdm import tqdm
from utils import seed_everything

def set_config():
    parser = argparse.ArgumentParser(description="联合消歧")
    parser.add_argument('--project_name',type=str,required=False, default='联合消歧',help='a name of this project')
    parser.add_argument('--device', type=int, required=False, default=0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--train_data', type=str, default='semcor', choices=['semcor', 'all'])
    parser.add_argument('--generate_tokens', action='store_true', default=True)
    parser.add_argument('--gloss_bsz', type=int, default=20, choices=[10,20, 60])
    parser.add_argument('--image_bsz', type=int, default=100, choices=[50,100, 150])

    args = parser.parse_args()
    return args

def vwsd_sentence_tokenization(tokenize,target_words,sentences,sent):
    max_length = 0
    s_tokens = []
    s_mask = []
    import nltk
    # if len(nltk.word_tokenize(target_words[0])) > 1:
    #     print()
    sent_without_target = sent.strip(target_words[0])
    sent_without_target = nltk.word_tokenize(sent_without_target)

    sent_tokens = []
    sent_tokens.extend(torch.tensor([49406], dtype=torch.int32))  # CLS TOKEN
    sent_mask = tuple()
    w_token = tokenize(target_words[0])[0]
    w_token = w_token[w_token != 0][1:-1]
    sent_mask = (len(sent_tokens), len(sent_tokens) + len(w_token))
    sent_tokens.extend(w_token)
    for w in sent_without_target:
        w_token = tokenize(w)[0]
        w_token = w_token[w_token != 0][1:-1]
        sent_tokens.extend(w_token)
    sent_tokens.extend(torch.tensor([49407], dtype=torch.int32))  # SEP TOKEN
    s_tokens.append(sent_tokens)
    s_mask.append(sent_mask)
    if len(sent_tokens) > max_length:
        max_length = len(sent_tokens)

    # for target_word, sentence in zip(target_words, sentences):
    #     sent_tokens = []
    #     sent_tokens.extend(torch.tensor([49406], dtype=torch.int32))  # CLS TOKEN
    #     sent_mask = tuple()
    #     for w in sentence:
    #         w_token = tokenize(w)[0]
    #         w_token = w_token[w_token != 0][1:-1]
    #         # if w == target_word:
    #         #     sent_mask = (len(sent_tokens),len(sent_tokens) + len(w_token))
    #         sent_tokens.extend(w_token)
    #     sent_tokens.extend(torch.tensor([49407], dtype=torch.int32))  # SEP TOKEN
    #     s_tokens.append(sent_tokens)
    #     s_mask.append(sent_mask)
    #     if len(sent_tokens) > max_length:
    #         max_length = len(sent_tokens)

    max_length = 77
    # PADDING
    for i in range(len(s_tokens)):
        if len(s_tokens[i]) < max_length:
            for _ in range(max_length - len(s_tokens[i])):
                s_tokens[i].extend(torch.tensor([0], dtype=torch.int32))
        elif len(s_tokens[i]) > max_length:
            if s_mask[i][1] <= 77:
                s_tokens[i] = s_tokens[i][:77]
            elif s_mask[i][1] > 77:
                new_tokens = s_tokens[i][s_mask[i][0]:s_mask[i][1]]  # center
                left_p = s_mask[i][0] - 1
                right_p = s_mask[i][1]
                new_length = len(new_tokens)
                while new_length < 77:
                    new_tokens = [s_tokens[i][left_p]] + new_tokens
                    left_p -= 1
                    new_length += 1
                    if new_length < 77:
                        if right_p < len(s_tokens[i]):
                            new_tokens = new_tokens + [s_tokens[i][right_p]]
                            right_p += 1
                            new_length += 1
                    elif new_length == 77:
                        break
                new_begin = s_mask[i][0] - (left_p + 1)
                new_end = s_mask[i][1] - (left_p + 1)
                s_mask[i] = (new_begin, new_end)
                s_tokens[i] = new_tokens

        s_tokens[i] = torch.vstack(s_tokens[i]).T
    s_tokens = torch.vstack(s_tokens)
    return s_tokens, s_mask

def sentence_tokenization(tokenize,target_words,sentences):
    max_length = 0
    s_tokens = []
    s_mask = []
    for target_word,sentence in zip(target_words,sentences):
        sent_tokens = []
        sent_tokens.extend(torch.tensor([49406],dtype=torch.int32)) #CLS TOKEN
        sent_mask = tuple()
        for w in sentence:
            w_token = tokenize(w)[0]
            w_token = w_token[w_token != 0][1:-1]
            if w == target_word:
                sent_mask = (len(sent_tokens),len(sent_tokens) + len(w_token))
            sent_tokens.extend(w_token)
        sent_tokens.extend(torch.tensor([49407],dtype=torch.int32)) #SEP TOKEN
        s_tokens.append(sent_tokens)
        s_mask.append(sent_mask)
        if len(sent_tokens) > max_length:
            max_length = len(sent_tokens)

    max_length = 77
    # PADDING
    for i in range(len(s_tokens)):
        if len(s_tokens[i]) < max_length:
            for _ in range(max_length-len(s_tokens[i])):
                s_tokens[i].extend(torch.tensor([0],dtype=torch.int32))
        elif len(s_tokens[i]) > max_length:
            if s_mask[i][1] <= 77:
                s_tokens[i] = s_tokens[i][:77]
            elif s_mask[i][1] > 77:
                new_tokens = s_tokens[i][s_mask[i][0]:s_mask[i][1]] #center
                left_p = s_mask[i][0] - 1
                right_p = s_mask[i][1]
                new_length = len(new_tokens)
                while new_length < 77:
                    new_tokens = [s_tokens[i][left_p]] + new_tokens
                    left_p -= 1
                    new_length += 1
                    if new_length < 77 :
                        if right_p < len(s_tokens[i]):
                            new_tokens =  new_tokens + [s_tokens[i][right_p]]
                            right_p += 1
                            new_length += 1
                    elif new_length == 77:
                        break
                new_begin = s_mask[i][0] - (left_p + 1)
                new_end = s_mask[i][1] - (left_p + 1)
                s_mask[i] = (new_begin,new_end)
                s_tokens[i] = new_tokens

        s_tokens[i] = torch.vstack(s_tokens[i]).T
    s_tokens = torch.vstack(s_tokens)
    return s_tokens,s_mask

def gloss_tokenization(tokenize,total_candidate_gloss):
    max_length = 0
    s_tokens = []
    for gloss in total_candidate_gloss:
        words = gloss.split(' ')
        sent_tokens = []
        sent_tokens.extend(torch.tensor([49406], dtype=torch.int32))  # CLS TOKEN
        for w in words:
            w_token = tokenize(w)[0]
            w_token = w_token[w_token != 0][1:-1]
            sent_tokens.extend(w_token)
        sent_tokens.extend(torch.tensor([49407],dtype=torch.int32)) #SEP TOKEN
        s_tokens.append(sent_tokens)
        if len(sent_tokens) > max_length:
            max_length = len(sent_tokens)

    max_length = 77
    # PADDING
    for i in range(len(s_tokens)):
        if len(s_tokens[i]) < max_length:
            for _ in range(max_length - len(s_tokens[i])):
                s_tokens[i].extend(torch.tensor([0], dtype=torch.int32))
        elif len(s_tokens[i]) > max_length:
            s_tokens[i] = s_tokens[i][:77]

        s_tokens[i] = torch.vstack(s_tokens[i]).T
    s_tokens = torch.vstack(s_tokens)
    return s_tokens

def image_path_transfer(d,image_dir_path,vwsd_image_dir):
    temp_dict = {}
    for k,v in d['candidate_image_path'].items():
        temp_list = []
        for path in v:
            if 'pkl' in path:
                new_path = os.path.join(vwsd_image_dir, path.split('\\')[-1])
            else:
                new_path = os.path.join(image_dir_path,path.split('\\')[-1])
            temp_list.append(new_path)
        temp_dict[k] = temp_list
    return temp_dict

# only execute one time
def get_all_tokens(file,tokenize):
    tokens_data = {}
    for d in tqdm(file):
        ### tokenize pre_process
        sentence_tokens, sentence_mask = vwsd_sentence_tokenization(tokenize, [d['word']], [d['words']],d['sentence'])
        gloss_tokens = gloss_tokenization(tokenize, d['candidate_gloss'])

        name = d['word'] + '#' + d['token_id']
        tokens_data[name] = {
            'sentence_tokens': sentence_tokens.tolist(),
            'sentence_mask': sentence_mask,
            'gloss_labels': d['candidate_labels'],
            'gloss_tokens': gloss_tokens.tolist(),
        }
    with open('vwsd_tokens.json','w',encoding='utf-8') as f:
        json.dump(tokens_data,f,indent=3,ensure_ascii=False)
    exit()
    
from torch.utils.data import Dataset,DataLoader
class shuffle_dataset(Dataset):
    def __init__(self,data):
        super(shuffle_dataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def collate_train_fn(batch):
    return batch[0]

def get_train_data(args,file,tokenize,image_dir,vwsd_image_dir):
    if 'vwsd' in args.train_path:
        filename = f'/home/featurize/data/vwsd_training_tokens.{str(args.gloss_bsz)}.{str(args.image_bsz)}.json'
    else:
        filename = f'/home/featurize/data/training_tokens.{str(args.gloss_bsz)}.{str(args.image_bsz)}.json'
    if os.path.exists(filename):
        print(filename,' 已经存在！')
    else:
        if 'vwsd' in args.train_path:
            tokens_data = json.load(open(r'/home/featurize/work/vwsd_tokens.json','r',encoding='utf-8'))
        else:
            tokens_data = json.load(open(r'/home/featurize/work/all_tokens.json','r',encoding='utf-8'))

        gloss_num = 0
        image_num = 0
        temp_data = []
        temp_record = []
        
        data = shuffle_dataset(file)
        data_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0,pin_memory=True, collate_fn=collate_train_fn)

        for d in data_loader:
            d['candidate_image_path'] = image_path_transfer(d,image_dir_path=image_dir,vwsd_image_dir=vwsd_image_dir)
            if d['gold_labels'][0] not in d['candidate_labels']:
                label_idx = d['candidate_gloss'].index(d['gold_gloss'][0])
                d['gold_labels'][0] = d['candidate_labels'][label_idx]
            candidate_gloss = d['candidate_gloss']
            candidate_image = [imgs for _, imgs in d['candidate_image_path'].items()]

            if len(candidate_gloss) <= args.gloss_bsz:
                if gloss_num + len(candidate_gloss) <= args.gloss_bsz and image_num + len(
                        candidate_image) <= args.image_bsz:
                    gloss_num += len(candidate_gloss)
                    image_num += len(candidate_image)
                    temp_record.append(d)
                else:
                    temp_data.append(temp_record)
                    temp_record = [d]
                    gloss_num = len(candidate_gloss)
                    image_num = len(candidate_gloss)
            else:
                if len(temp_record) != 0:
                    temp_data.append(temp_record)
                    temp_record = []
                    gloss_num = 0
                    image_num = 0

                sample_candidate_labels = sample(d['candidate_labels'], args.gloss_bsz)
                while d['gold_labels'][0] not in sample_candidate_labels:
                    sample_candidate_labels = sample(d['candidate_labels'], args.gloss_bsz)
                else:
                    sample_candidate_gloss = []
                    sample_candidate_image_path = {}
                    for sample_label in sample_candidate_labels:
                        sample_candidate_gloss.append(d['candidate_gloss'][d['candidate_labels'].index(sample_label)])
                        sample_candidate_image_path[sample_label] = d['candidate_image_path'][sample_label]
                    d['candidate_labels'] = sample_candidate_labels
                    d['candidate_gloss'] = sample_candidate_gloss
                    d['candidate_image_path'] = sample_candidate_image_path
                    temp_data.append([d])

        data = []
        training_tokens = {}
        add_num = 0
        for id, l in enumerate(tqdm(temp_data[add_num:])):
            temp_word = []
            temp_word_list = []
            temp_sentence = []
            temp_gold_labels = []
            temp_gold_gloss = []
            temp_candidate_gloss_labels = []
            temp_candidate_gloss = []
            total_candidate_gloss = []
            temp_candidate_image_labels = []
            temp_candidate_image = []
            total_candidate_image = []
            gold_count = 0
            image_count = 0
            for d in l:
                temp_word.append(d['word'])
                temp_word_list.append(d['words'])
                temp_sentence.append(d['sentence'])
                gold_idx = d['candidate_labels'].index(d['gold_labels'][0])
                temp_gold_labels.append(gold_idx)
                temp_gold_gloss.append(d['gold_gloss'][0])
                temp_candidate_gloss_labels.append((gold_count, gold_count + len(d['candidate_labels'])))
                temp_candidate_gloss.append(d['candidate_gloss'])
                total_candidate_gloss += d['candidate_gloss']
                gold_count += len(d['candidate_labels'])
                for _, imgs_list in d['candidate_image_path'].items():
                    temp_candidate_image_labels.append((image_count, image_count + len(imgs_list)))
                    image_count += len(imgs_list)
                    temp_candidate_image.append(imgs_list)
                    total_candidate_image += imgs_list

            image2gloss_labels = []
            for idx, tup in enumerate(temp_candidate_image_labels):
                image2gloss_labels += [idx for i in range(tup[0], tup[1])]

            sentence2image_labels = []
            for label, cand_label in zip(temp_gold_labels, temp_candidate_gloss_labels):
                sentence2image_labels.append(temp_candidate_image_labels[cand_label[0]:cand_label[1]][label])

            ### tokenize pre_process
            # sentence_tokens2, sentence_mask2 = sentence_tokenization(tokenize, temp_word, temp_word_list)
            # gloss_tokens2 = gloss_tokenization(tokenize, total_candidate_gloss)
            # sentence_tokens2 = sentence_tokens2.tolist()
            # gloss_tokens2 = gloss_tokens2.tolist()
            # training_tokens[id] = {
            #     'sentence_tokens': sentence_tokens.tolist(),
            #     'sentence_mask': sentence_mask,
            #     'gloss_tokens': gloss_tokens.tolist()
            # }
            sentence_tokens, sentence_mask, gloss_tokens = [],[],[]
            for d in l:
                name = d['word'] + '#' + d['token_id']
                content = tokens_data.get(name)
                sentence_tokens.append(content['sentence_tokens'][0])
                sentence_mask.append(content['sentence_mask'][0])
                for bn_id in d['candidate_labels']:
                    index = content['gloss_labels'].index(bn_id)
                    gloss_tokens.append(content['gloss_tokens'][index])

            assert len(sentence_tokens) == len(sentence_mask)
            assert len(sentence_tokens) > 0
            assert len(gloss_tokens) > 0

            data.append({
                'id': id,
                'word': temp_word,
                'word_list': temp_word_list,
                'sentence': temp_sentence,

                'sentence2gloss_labels': temp_gold_labels,
                'gold_gloss': temp_gold_gloss,
                'candidate_gloss_labels': temp_candidate_gloss_labels,
                'candidate_gloss': temp_candidate_gloss,
                'total_candidate_gloss': total_candidate_gloss,

                'gloss2image_labels': temp_candidate_image_labels,
                'candidate_image': temp_candidate_image,
                'total_candidate_image': total_candidate_image,

                'image2gloss_labels': image2gloss_labels,
                'sentence2image_labels': sentence2image_labels,

                'sentence_tokens': sentence_tokens,
                'sentence_mask': sentence_mask,
                'gloss_tokens': gloss_tokens
            })

        # generate pre-processed tokens.json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=3, ensure_ascii=False)
        print(filename,'新的训练数据已生成')

def get_test_data(args, file, tokenize):
    tokens_data = []
    for d in tqdm(file):
        ### tokenize pre_process
        sentence_tokens, sentence_mask = sentence_tokenization(tokenize, [d['word']], [d['words']])
        gloss_tokens = gloss_tokenization(tokenize, d['candidate_gloss'])

        assert len(sentence_tokens) == len(sentence_mask)
        assert len(sentence_tokens) > 0
        assert len(gloss_tokens) > 0

        d['sentence2gloss_labels'] = [d['candidate_labels'].index(d['gold_labels'][0])]
        d['total_candidate_gloss'] = d['candidate_gloss']
        d['candidate_gloss_labels'] = [[0,len(d['candidate_gloss'])]]

        d['sentence_tokens']= sentence_tokens.tolist(),
        d['sentence_mask']= sentence_mask,
        d['gloss_tokens']= gloss_tokens.tolist(),

        if 'senseval2' in d['token_id']:
            d['class'] = '02'
        elif 'senseval3' in d['token_id']:
            d['class'] = '03'
        elif 'semeval2010' in d['token_id']:
            d['class'] = '10'
        elif 'semeval2013' in d['token_id']:
            d['class'] = '13'
        elif 'semeval2015' in d['token_id']:
            d['class'] = '15'
        else:
            d['class'] = '07'

        tokens_data.append(d)

    with open('all_test_data.json', 'w', encoding='utf-8') as f:
        json.dump(tokens_data, f, indent=3, ensure_ascii=False)
    exit()

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("Need available GPU(s) to run this model...")
        quit()

    # parse args
    args = set_config()
    args.gloss_bsz = 100
    args.image_bsz = args.gloss_bsz * 5
    args.model_name = 'ViT-L/14'

    seed_everything(args.seed)
    # args.train_path = r'/home/featurize/data/vwsd_total_training_data2.json'
    args.train_path = r'/home/featurize/data/semcor_training_data.json'
    model = mymodel(args)

    image_dir = r'/home/featurize/data/image_pt'
    vwsd_image_dir = r'/home/featurize/data/L-VWSD/image_vecs'
    print(args.train_path)
    # get_all_tokens(json.load(open(train_path, 'r', encoding='utf-8')), model.tokenize)
    # get_test_data = get_test_data(args, json.load(open(r'E:\联合消歧工作\形成训练集\test_data\all_test_data.json', 'r', encoding='utf-8')), model.tokenize)
    train_data = get_train_data(args, json.load(open(args.train_path, 'r', encoding='utf-8')),model.tokenize,image_dir,vwsd_image_dir)