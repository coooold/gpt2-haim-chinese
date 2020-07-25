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
        self.pad_token_id = 0
        self.n_ctx = n_ctx
        self.stride = stride
        self.tokenizer = tokenizer

        self.train_data = []
        self.train_data_index = []
        self.train_data_count = 0

        self.tokenizer = tokenizer
        self.sep_token_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize('。，；？！[SEP]')
        )
        index = 0
        for f in self.scan_files(tokenized_file_path):
            print("load training {}".format(f))
            raw = np.fromfile(f, dtype=np.int16).tolist()
            sample_count = (len(raw) - self.n_ctx) // self.stride
            index += sample_count
            self.train_data.append(raw)
            self.train_data_index.append(index)

    def __len__(self):
        return self.train_data_index[len(self.train_data_index) - 1]

    def __getitem__(self, i):
        # 找到i在哪个文件中
        j = 0
        for idx, num in enumerate(self.train_data_index):
            if i > num:
                break
            j = idx
        raw = self.train_data[j]
        pos = i - self.train_data_index[j-1] if j > 0 else 0

        sample = raw[self.stride * pos: self.stride * pos + self.n_ctx]
        return self.process_sample(sample)

    def to_sentences(self, s):
        sentences = []
        sentence = []
        for i, t in enumerate(s):
            sentence.append(t)
            if self.sep_token_ids.__contains__(t):
                sentences.append(sentence)
                sentence = []
        return sentences

    @staticmethod
    def scan_files(tokenized_data_path: str):
        train_files = []
        for root, subdirs, files in os.walk(tokenized_data_path):
            for file in files:
                train_files.append(tokenized_data_path + '/' + file)
        shuffle(train_files)
        return train_files

    def process_sample(self, arr):
        # print("loading tokenized file: {}".format(tokenized_file))

        min_len = 32
        max_suffix_sentences = 3
        sample_len = randint(min_len, self.n_ctx - 6)
        sample = arr[:sample_len]

        sentences = self.to_sentences(sample)
        sentences_len = len(sentences)

        if sentences_len == 1:
            suffix_sentence_count = 0
        else:
            suffix_sentence_count = randint(1, min(len(sentences) - 1, max_suffix_sentences))
        prefix_sentence_count = len(sentences) - suffix_sentence_count

        prefix_sentences = sentences[:prefix_sentence_count]
        suffix_sentences = sentences[prefix_sentence_count:]

        prefix_tokens = []
        for i in prefix_sentences:
            prefix_tokens.extend(i)
        suffix_tokens = []
        for i in suffix_sentences:
            suffix_tokens.extend(i)

        # if there is a suffix
        if len(suffix_tokens) > 0:
            suffix_tokens.append(self.begin_token_id)
        prefix_tokens.append(self.end_token_id)
        suffix_len = len(suffix_tokens)
        prefix_len = len(prefix_tokens)

        """
        # The random shift \deltaδ is introduced to soften the length constraint, 
        effectively allowing the model some leeway at inference time. 
        We sampled the position shift uniformly in \left[0, 0.1\times n\right][0,0.1×n].
        """
        delta = int((random() - 0.5) * 7)  # [-3, 3]
        suffix_positions = [prefix_len + 1 + i + delta for i in range(suffix_len)]
        prefix_positions = [i for i in range(len(prefix_tokens))]

        # All labels set to ``-100`` are ignored (masked), the loss is only
        #             computed for labels in ``[0, ..., config.vocab_size]``
        # modeling_gpt2.py
        final_sample = suffix_tokens + prefix_tokens
        final_sample_positions = suffix_positions + prefix_positions
        final_label = [-100] * suffix_len + prefix_tokens

        # padding
        if len(final_sample) < self.n_ctx:
            final_sample = final_sample + [self.pad_token_id] * (self.n_ctx - len(final_sample))
            final_sample_positions = final_sample_positions + [0] * (self.n_ctx - len(final_sample_positions))
            final_label = final_label + [-100] * (self.n_ctx - len(final_label))

        return {
            'input_ids': final_sample,
            'labels': final_label,
            'position_ids': final_sample_positions,
        }
