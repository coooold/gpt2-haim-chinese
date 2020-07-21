#!/usr/bin/env python
# coding=utf-8
import warnings
import os
import logging
from .model_args import ModelArguments, parse_args
from .dataset import GPT2Dataset
from .tokenization_bert import BertTokenizer
import torch

# 屏蔽tensorflow无效日志信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
