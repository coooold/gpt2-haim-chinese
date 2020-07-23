#!/usr/bin/env python
# coding=utf-8
from torch.utils.data.dataset import Dataset
import numpy as np
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
        self.labels = []

        for f in self.scan_files(tokenized_file_path):
            self.load_samples(f)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return {
            'input_ids': self.features[i],
            'labels': self.labels[i],
            'position_ids': self.positions[i],
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
            sample_len = randint(min_len, self.n_ctx - 6)
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
            delta = int((random() - 0.5) * 7)  # [-3, 3]
            suffix = sample[prefix_len:sample_len] + [self.begin_token_id]
            suffix_len = len(suffix)
            suffix_positions = [prefix_len + 1 + i + delta for i in range(suffix_len)]
            prefix = sample[0:prefix_len] + [self.end_token_id]
            prefix_positions = [i for i in range(len(prefix))]

            # All labels set to ``-100`` are ignored (masked), the loss is only
            #             computed for labels in ``[0, ..., config.vocab_size]``
            # modeling_gpt2.py
            final_sample = suffix + prefix
            final_sample_positions = suffix_positions + prefix_positions
            final_label = [-100] * suffix_len + prefix

            # padding
            if len(final_sample) < self.n_ctx:
                final_sample = final_sample + [self.pad_token_id] * (self.n_ctx - len(final_sample))
                final_sample_positions = final_sample_positions + [0] * (self.n_ctx - len(final_sample_positions))
                final_label = final_label + [-100] * (self.n_ctx - len(final_label))

            self.features.append(final_sample)
            self.positions.append(final_sample_positions)
            self.labels.append(final_label)

            start_point += self.stride
