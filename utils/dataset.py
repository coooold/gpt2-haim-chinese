#!/usr/bin/env python
# coding=utf-8
from torch.utils.data.dataset import Dataset
import numpy as np
import torch
from random import random, randint, shuffle
import math
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
        self.pad_token_id = 0
        self.n_ctx = n_ctx
        self.stride = stride
        self.tokenizer = tokenizer

        self.features = []
        self.positions = []
        self.suffixs = []
        self.suffix_positions = []

        for f in self.scan_files(tokenized_file_path):
            self.load_samples(f)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        sample = torch.tensor(self.features[i]).long()
        positions = torch.tensor(self.positions[i]).long()
        suffix = torch.tensor(self.suffixs[i]).long()
        suffix_positions = torch.tensor(self.suffix_positions[i]).long()

        return {
            'input_ids': sample,
            'position_ids': positions,
            'labels': sample,
            'suffix_ids': suffix,
            'suffix_position_ids': suffix_positions,
        }

    @staticmethod
    def scan_files(tokenized_data_path: str):
        train_files = []
        for root, subdirs, files in os.walk(tokenized_data_path):
            for file in files:
                train_files.append(tokenized_data_path + '/' + file)
        shuffle(train_files)
        return train_files[0:7]  # 只装载几个文件，防止爆内存

    def load_samples(self, tokenized_file):
        print("loading tokenized file: {}".format(tokenized_file))

        arr = np.fromfile(tokenized_file, dtype=np.int16).tolist()
        arr_len = len(arr)
        min_len = 32

        start_point = 0
        # 按滑动窗口切割内容
        while start_point + self.n_ctx < arr_len:
            sample_len = randint(min_len, self.n_ctx - 2)
            sample = arr[start_point: start_point + sample_len]
            # 句子切分为 prefix_len + suffix_len
            prefix_len = math.floor(
                sample_len * (random() * 0.3 + 0.5)
            )

            """
            # The random shift \deltaδ is introduced to soften the length constraint, 
            effectively allowing the model some leeway at inference time. 
            We sampled the position shift uniformly in \left[0, 0.1\times n\right][0,0.1×n].
            """
            # delta = randint(0, math.floor(0.1 * sample_len))
            suffix = sample[prefix_len:sample_len] + [self.begin_token_id]
            suffix_positions = [prefix_len + 1 + i for i in range(len(suffix))]
            prefix = sample[0:prefix_len] + [self.end_token_id]
            prefix_positions = [i for i in range(len(prefix))]

            self.suffixs.append(suffix)
            self.suffix_positions.append(suffix_positions)
            self.features.append(prefix)
            self.positions.append(prefix_positions)

            start_point += self.stride
