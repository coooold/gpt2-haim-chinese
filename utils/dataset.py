#!/usr/bin/env python
# coding=utf-8
from torch.utils.data.dataset import Dataset
import numpy as np
from random import random, randint, shuffle
import os


class GPT2Dataset(Dataset):
    def __init__(self,
                 n_ctx,
                 stride,
                 tokenized_file_path,
                 tokenizer
                 ):
        self.begin_token_id = tokenizer.convert_tokens_to_ids('<begin>')
        self.end_token_id = tokenizer.convert_tokens_to_ids('<end>')
        self.cls_token_id = tokenizer.convert_tokens_to_ids('[CLS]')
        self.pad_token_id = 0
        self.n_ctx = n_ctx
        self.stride = stride
        self.tokenizer = tokenizer

        self.data = []
        for f in self.scan_files(tokenized_file_path):
            print("load training {}".format(f))
            raw = np.fromfile(f, dtype=np.int16).tolist()
            self.data += raw + [self.cls_token_id]

    def __len__(self):
        return (len(self.data) - self.n_ctx) // self.stride

    def __getitem__(self, i):
        pos = self.stride * i
        sample = self.data[pos: pos + self.n_ctx - 2]
        return self.process_sample(sample)

    @staticmethod
    def scan_files(tokenized_data_path: str):
        train_files = []
        for root, subdirs, files in os.walk(tokenized_data_path):
            for file in files:
                train_files.append(tokenized_data_path + '/' + file)
        # shuffle(train_files)
        return train_files

    def process_sample(self, arr):
        sample_len = len(arr)
        prefix_len = sample_len - randint(1, int(sample_len * 0.3))
        prefix = arr[0:prefix_len]
        suffix = arr[prefix_len:sample_len]

        final_sample = suffix + [self.begin_token_id] + prefix + [self.end_token_id]

        return {
            'input_ids': final_sample,
            'labels': final_sample
        }
