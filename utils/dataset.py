#!/usr/bin/env python
# coding=utf-8
from torch.utils.data.dataset import Dataset
import numpy as np
import torch
from random import random
import math

"""
To condition the output on the suffix, we rearrange the input sequences such that the first ss tokens are the suffix, followed by the prefix, separated by \texttt{<begin>}<begin> and \texttt{<end>}<end> tokens. We found that in order for the model to properly “stitch” the generated text to the suffix, it is necessary to indicate the starting position of the suffix, thereby dictating the sequence length. We do this by assigning the suffix (prefix) tokens with positional embeddings corresponding to their original positions at the end (beginning) of the sequence, rather than their position in the rearranged sequence.
"""


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
        min_len = 32

        self.features = []
        self.positions = []

        start_point = 0
        # 按滑动窗口切割内容
        while start_point + n_ctx < arr_len:
            sample_len = random.randint(min, n_ctx - 2)
            sample = arr[start_point: start_point + sample_len]
            prefix_len = max(math.floor(n_ctx * 0.5 * random()), 3)
            suffix_len = math.floor(n_ctx * 0.3 * random())
            suffix_len = suffix_len if suffix_len > n_ctx * 0.03 else 0  # 10%的情况没有suffix
            middle_len = sample_len - prefix_len - suffix_len

            """
            # The random shift \deltaδ is introduced to soften the length constraint, 
            effectively allowing the model some leeway at inference time. 
            We sampled the position shift uniformly in \left[0, 0.1\times n\right][0,0.1×n].
            """
            delta = random.randint(0, int(0.1 * sample_len))
            suffix = sample[prefix_len + middle_len:sample_len] + [begin_token_id]
            suffix_positions = [i + prefix_len + 1 + middle_len + delta for i in range(len(suffix))]
            prefix = sample[0:prefix_len] + [end_token_id]
            prefix_positions = [i for i in range(len(prefix_len))]
            final_sample = suffix + prefix
            final_sample_positions = suffix_positions + prefix_positions

            self.features.append(final_sample)
            self.positions.append(final_sample_positions)

            start_point += stride

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        sample = torch.tensor(self.features[i]).long()
        positions = torch.tensor(self.positions[i]).long()
        return {
            'input_ids': sample,
            'labels': sample,
            'position_ids': positions
        }
