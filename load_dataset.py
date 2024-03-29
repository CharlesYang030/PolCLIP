import json
import os
import torch
from random import sample
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
import warnings
warnings.filterwarnings("ignore")

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

def image_path_transfer(d,image_dir_path):
    temp_dict = {}
    for k,v in d['candidate_image_path'].items():
        temp_list = []
        for path in v:
            new_path = os.path.join(image_dir_path,path.split('\\')[-1])
            temp_list.append(new_path)
        temp_dict[k] = temp_list
    return temp_dict

def get_data(args,file,tokenize):
    ## pre-processed tokens to tensor
    tokens = json.load(open('training_tokens2.json','r',encoding='utf-8'))
    for k, v in tokens.items():
        v['sentence_tokens'] = torch.tensor(v['sentence_tokens'], dtype=torch.int32)
        v['gloss_tokens'] = torch.tensor(v['gloss_tokens'], dtype=torch.int32)
    ##

    gloss_num = 0
    image_num = 0
    temp_data = []
    temp_record = []
    for d in file:
        # d['candidate_image_path'] = image_path_transfer(d,image_dir_path=r'/mnt/image_pt')
        candidate_gloss = d['candidate_gloss']
        candidate_image = [imgs for _,imgs in d['candidate_image_path'].items()]

        if len(candidate_gloss) <= args.gloss_bsz:
            if gloss_num + len(candidate_gloss) <= args.gloss_bsz and image_num + len(candidate_image) <= args.image_bsz:
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

            sample_candidate_labels = sample(d['candidate_labels'],args.gloss_bsz)
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
    for id,l in enumerate(tqdm(temp_data[add_num:])):
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
            temp_candidate_gloss_labels.append((gold_count,gold_count+len(d['candidate_labels'])))
            temp_candidate_gloss.append(d['candidate_gloss'])
            total_candidate_gloss += d['candidate_gloss']
            gold_count += len(d['candidate_labels'])
            for _,imgs_list in d['candidate_image_path'].items():
                temp_candidate_image_labels.append((image_count,image_count+len(imgs_list)))
                image_count += len(imgs_list)
                temp_candidate_image.append(imgs_list)
                total_candidate_image += imgs_list

        image2gloss_labels = []
        for idx,tup in enumerate(temp_candidate_image_labels):
            image2gloss_labels += [idx for i in range(tup[0],tup[1])]

        sentence2image_labels = []
        for label,cand_label in zip(temp_gold_labels,temp_candidate_gloss_labels):
            sentence2image_labels.append(temp_candidate_image_labels[cand_label[0]:cand_label[1]][label])

        # if args.generate_tokens:
        #     ### tokenize pre_process
        #     sentence_tokens,sentence_mask = sentence_tokenization(tokenize,temp_word,temp_word_list)
        #     gloss_tokens = gloss_tokenization(tokenize,total_candidate_gloss)
        #     training_tokens[id] = {
        #         'sentence_tokens': sentence_tokens.tolist(),
        #         'sentence_mask': sentence_mask,
        #         'gloss_tokens': gloss_tokens.tolist()
        #     }

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

            'sentence_tokens': tokens.get(str(id+add_num))['sentence_tokens'],
            'sentence_mask': tokens.get(str(id+add_num))['sentence_mask'],
            'gloss_tokens': tokens.get(str(id+add_num))['gloss_tokens']
        })

    # # if args.generate_tokens:
    #     generate pre-processed tokens.json
    #     # with open('training_tokens.10.50_all.json','w',encoding='utf-8') as f:
    #     #     json.dump(training_tokens,f,indent=3,ensure_ascii=False)
    #     # exit()
    return data

class train_dataset(Dataset):
    def __init__(self, data,args, mode):
        super(train_dataset, self).__init__()
        self.data = data
        self.args = args
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # self.data[item]['sentence_tokens'] = torch.tensor(self.data[item]['sentence_tokens'], dtype=torch.int32)
        # self.data[item]['gloss_tokens'] = torch.tensor(self.data[item]['gloss_tokens'], dtype=torch.int32)
        # self.data[item]['images_tokens'] = torch.vstack([torch.load(x).unsqueeze(0) for x in self.data[item]['total_candidate_image']])

        # id = self.data[item]['id']
        # word = self.data[item]['word']
        # word_list = self.data[item]['word_list']
        # sentence = self.data[item]['sentence']
        # sentence2gloss_labels = self.data[item]['sentence2gloss_labels']
        # gold_gloss = self.data[item]['gold_gloss']
        # candidate_gloss_labels = self.data[item]['candidate_gloss_labels']
        # candidate_gloss = self.data[item]['candidate_gloss']
        # total_candidate_gloss = self.data[item]['total_candidate_gloss']
        # gloss2image_labels = self.data[item]['gloss2image_labels']
        # candidate_image = self.data[item]['candidate_image']
        # total_candidate_image = self.data[item]['total_candidate_image']
        # image2gloss_labels = self.data[item]['image2gloss_labels']
        # sentence2image_labels = self.data[item]['sentence2image_labels']
        # sentence_tokens = self.data[item]['sentence_tokens']
        # sentence_mask = self.data[item]['sentence_mask']
        # gloss_tokens = self.data[item]['gloss_tokens']
#         try:
#             self.data[item]['images_tokens'] = torch.vstack([torch.load(x).unsqueeze(0) for x in self.data[item]['total_candidate_image']])
#         except:
#             self.data[item]['images_tokens'] = torch.vstack([get_img_vec(x) for x in self.data[item]['total_candidate_image']])
        
        return self.data[item]
        # return id,word,word_list,sentence,sentence2gloss_labels,gold_gloss,candidate_gloss_labels,candidate_gloss,total_candidate_gloss,\
        #        gloss2image_labels,candidate_image,total_candidate_image,image2gloss_labels,sentence2image_labels,sentence_tokens,sentence_mask,gloss_tokens


def collate_train_fn(batch):
    return batch

class eval_dataset(Dataset):
    def __init__(self, data,args, mode):
        super(eval_dataset, self).__init__()
        self.data = data
        self.args = args
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # self.data[item]['sentence_tokens'] = torch.tensor(self.data[item]['sentence_tokens'][0], dtype=torch.int32)
        # self.data[item]['sentence_mask'] = self.data[item]['sentence_mask'][0]
        # self.data[item]['gloss_tokens'] = torch.tensor(self.data[item]['gloss_tokens'][0], dtype=torch.int32)

        return self.data[item]

def get_dataloader(args,data,mode):
    if mode =='train':
        mydataset = train_dataset(data, args=args, mode=mode)
        train_sampler = torch.utils.data.distributed.DistributedSampler(mydataset)
        data_loader = torch.utils.data.DataLoader(
            mydataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=train_sampler,
            collate_fn=collate_train_fn)
        # data_loader = DataLoader(mydataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True,collate_fn=collate_train_fn)
    elif mode == 'eval':
        mydataset = eval_dataset(data, args=args, mode=mode)
        data_loader = DataLoader(mydataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True,collate_fn=collate_train_fn)
    return data_loader

class vwsd_dataset(Dataset):
    def __init__(self, data,args):
        super(vwsd_dataset, self).__init__()
        self.data = data
        self.args = args

    def __len__(self):
        return len(self.data)

    def get_img_vec(self,image_path):
        import pickle
        with open(image_path, 'rb') as f:
            image_vec = pickle.load(f)
            key = list(image_vec.keys())[0]
            image_vec = image_vec[key].unsqueeze(0)
        return image_vec

    def __getitem__(self, item):
        amb = self.data[item]['amb']
        phrase = self.data[item]['phrase']
        sense = self.data[item]['sense']
        img_paths = self.data[item]['img_paths']
        for i,path in enumerate(img_paths):
            path = os.path.join('/home/featurize/data/image_vecs',path.split('\\')[-1])
            img_paths[i] = path
        sentence = 'A photo of "' + phrase + '", ' + sense.lower()
        # sentence = 'A photo of "' + phrase + '"'
        candidate_images_vec = torch.vstack([self.get_img_vec(path) for path in img_paths])

        self.data[item]['sentence'] = sentence
        self.data[item]['candidate_images_vec'] = candidate_images_vec

        return self.data[item]

def get_vwsd_dataloader(args,vwsd_data):
    mydataset = vwsd_dataset(vwsd_data, args=args)
    data_loader = DataLoader(mydataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,collate_fn=collate_train_fn)
    return data_loader