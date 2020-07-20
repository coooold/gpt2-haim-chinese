#!/usr/bin/env python
# coding=utf-8
from dataclasses import dataclass, field
from typing import List, Optional
from transformers import HfArgumentParser, TrainingArguments


def parse_args():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    return parser.parse_args_into_dataclasses()


@dataclass()
class ModelArguments:
    """
    model arguments
    """
    model_config_file: str = field(metadata={"help": "path to model config json"})
    stride: int = field(metadata={"help": "stride window"})
    data_dir: str = field(metadata={"help": "tokenized data path"})
    vocab_file: str = field(metadata={"help": "tokenize_file"})
    pretrained_model_path: Optional[str] = field(default=None, metadata={"help": "pretrained model file"})
