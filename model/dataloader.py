import os, sys
import glob
import time

import numpy as np
import torch
import json
import nltk
import argparse
import fnmatch
import random

from transformers import BertTokenizer



class Dataset():

    def __init__(self):
        self.article = None
        self.ph = []
        self.ops = []
        self.ans = []
    

    def __convert_tokens_to_ids(self, tokenizer):
        self.article = tokenizer.convert_tokens_to_ids(self.article)
        self.article = torch.Tensor(self.article)
        for i in range(len(self.ops)):
            for k in range(4):
                self.ops[i][k] = tokenizer.convert_tokens_to_ids(self.ops[i][k])
                self.ops[i][k] = torch.Tensor(self.ops[i][k])
        self.ph = torch.Tensor(self.ph)
        self.ans = torch.Tensor(self.ans)


class Preprocess():

    def __init__(self, args, device = "cuda"):
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        self.data_dir = args.data_dir
        self.list_file = self.get_file(args.data_dir)

        self.data = []
        self.max_article_length = 0
        for list_file in self.list_file:
            data = json.load(open(list_file, 'r'))
            self.data.append(data)
            self.max_article_length = max(self.max_article_length, len(nltk.word_tokenize(data['article'])))
        
        self.data_objs = []

        for sample in self.data:
            self.data_objs += self._create_sample(sample)
        
        for i in range(len(self.data_objs)):
            self.data_objs[i].__convert_tokens_to_ids(self.tokenizer)
        torch.save(self.data_objs, args.save_name)

    def _create_sample(self, data):
        cnt = 0
        article = self.tokenizer.tokenize(data['article'])

        if len(article) > (2 ** 10):
            print("Length of article > 1024")
        
        if len(article) > (2 ** 9):
            sample = Dataset()
            sample.article = article
            for p in range(len(article)):
                if sample.article[p] == "_":
                    sample.article[p] = "[MASK]"
                    sample.ph.append(p)
                    ops = self.tokenize_ops(data['options'][cnt], self.tokenizer)
                    sample.ops.append(ops)
                    sample.ans.append(ord(data["answers"][cnt]) - ord('A'))
                    cnt += 1
            return [sample]
        else:
            first_sample = Dataset()
            second_sample = Dataset()
            second_s = len(article) - 512
            for p in range(len(article)):
                if (article[p] == '_'):
                    article[p] = '[MASK]'
                    ops = self.tokenize_ops(data['options'][cnt], self.tokenizer)
                    if (p < 512):
                        first_sample.ph.append(p)
                        first_sample.ops.append(ops)
                        first_sample.ans.append(ord(data['answers'][cnt]) - ord('A'))
                    else:
                        second_sample.ph.append(p - second_s)
                        second_sample.ops.append(ops)
                        second_sample.ans.append(ord(data['answers'][cnt]) - ord('A'))
                    cnt += 1                    
            first_sample.article = article[:512]
            second_sample.article = article[-512:]
            if (len(second_sample.ans) == 0):
                return [first_sample]
            else:
                return [first_sample, second_sample]
    
    def tokenize_ops(self, ops, tokenizer):
        ret = []
        for i in range(4):
            ret.append(tokenizer.tokenize(ops[i]))
        return ret
    
    def get_file(self, data_dir):
        files = []
        for dir_names in os.listdir(data_dir):
            for file_name in os.listdir(dir_names):
                if file_name.endswith(".json"):
                    files.append(os.path.join(data_dir, dir_names, file_name))
        return files


class Dataloader(object):

    def __init__(self, data_dir, data_file, cache_size, batch_size, device = "cuda"):

        self.data_dir = os.path.join(data_dir, data_file)
        print("[INFO] Loading {}".format(self.data_dir))
        self.data = torch.load(self.data_dir)
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.data_num = len(self.data)
        self.device = device

    def __getdata__(self, data_set, data_batch):

        max_article_length = 0
        max_option_length = 0
        max_ops_num = 0
        bsz = len(data_batch)
        for idx in data_batch:
            data = data_set[idx]
            max_article_length = max(max_article_length, data.article.size(0))
            for ops in data.ops:
                for op in ops:
                    max_option_length = max(max_option_length, op.size(0))
                max_ops_num = max(max_ops_num, len(data.ops))
        articles = torch.zeros(bsz, max_article_length).long()
        articles_mask = torch.ones(articles.size())

        options = torch.zeros(bsz, max_ops_num, 4, max_option_length).long()
        options_mask = torch.ones(options.size())

        answers = torch.zeros(bsz, max_ops_num).long()
        answers_mask = torch.ones(answers.size())

        question_pos = torch.zeros(answers.size()).long()

        for i, idx in enumerate(data_batch):
            data = data_set[idx]
            articles[i, : data.article.size(0)] = data.article
            articles[i, data.article.size(0): ] = 0
            for q, ops in enumerate(data.ops):
                for k, op in enumerate(ops):
                    options[i, q, k, :op.size(0): ] = 0
                    options_mask[i, q, k, op.size(0): ] = 0
            for q, ans in enumerate(data.ans):
                answers[i, q] = ans
                answers_mask[i, q] = 1
            for q, pos in enumerate(data.ph):
                question_pos[i, q] = pos
            
        input_ids = [articles, articles_mask, options, options_mask, question_pos, answers_mask]
        target = answers

        return input_ids, target

    def to_device(self, L, device):
        if (type(L) != list):
            return L.to(device)
        else:
            ret = []
            for item in L:
                ret.append(self.to_device(item, device))
            return ret


    def __getitem__(self, shuffle = True):
        if shuffle:
            random.shuffle(self.data)
        
        seqlen = torch.zeros(self.data_num)
        for i in range(self.data_num):
            seqlen[i] = self.data[i].article.size(0)
        cache_start = 0
        while cache_start < self.data_num:
            cache_end = min(cache_start + self.cache_size, self.data_num)
            cache_data = self.data[cache_start : cache_end]
            seql       = seqlen[cache_start : cache_end]
            _, indices = torch.sort(seql, descending=True)
            batch_start = 0
            while (batch_start + cache_start < cache_end):
                batch_end = min(batch_start + self.batch_size, cache_end - cache_start)
                data_batch = indices[batch_start:batch_end]
                input_ids, target = self.__getdata__(cache_data, data_batch)
                input_ids = self.to_device(input_ids, self.device)
                target = self.to_device(target, self.device)
                yield input_ids, target
                batch_start += self.batch_size
            cache_start += self.cache_size


if __name__ == "__main__":

    data_dir = ""
    folder_model = "../pretrained"
    model_name = "bert-base-uncased"
    parser = argparse.ArgumentParser(description='BERT for FIT task')
    args = parser.parse_args()
    for item in ["train", "valid", "test"]:
        args.data_dir = os.path.join(args.data_dir, item)
        args.pre = args.post = 0
        args.bert_model = os.path.join(folder_model, model_name)
        args.save_name  = f'./data/{item}_{model_name}.pt'
        data = Preprocess(args)
    