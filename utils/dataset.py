#!/usr/bin/env python
# coding=utf-8
from torch.utils.data.dataset import Dataset
import numpy as np
import torch
from random import random
import math


class GPT2Dataset(Dataset):
    def __init__(self,
                 n_ctx,
                 stride,
                 tokenized_file,
                 tokenizer
                 ):
        begin_token_id = tokenizer.convert_tokens_to_ids('<begin>')
        end_token_id = tokenizer.convert_tokens_to_ids('<end>')

        arr = np.fromfile(tokenized_file, dtype=np.int16).tolist()
        arr_len = len(arr)
        self.features = []
        start_point = 0
        # 按滑动窗口切割内容
        while start_point + n_ctx < arr_len:
            sample = arr[start_point: start_point + n_ctx]
            sample_len = n_ctx
            prefix_len = max(math.floor(n_ctx * 0.4 * random()), 3)
            suffix_len = math.floor(n_ctx * 0.3 * random())
            suffix_len = suffix_len if suffix_len > n_ctx*0.03 else 0  # 10%的情况没有suffix
            middle_len = sample_len - prefix_len - suffix_len
            prefix = sample[0:prefix_len]
            middle = sample[prefix_len:prefix_len+middle_len]
            suffix = sample[prefix_len+middle_len:sample_len]
            final_sample = prefix + [begin_token_id] + middle + [end_token_id] + suffix
            final_sample = final_sample[0:n_ctx]

            self.features.append(final_sample)
            start_point += stride

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        sample = torch.tensor(self.features[i]).long()
        return {
            'input_ids': sample,
            'labels': sample
        }
