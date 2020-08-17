#!/usr/bin/env python
# coding=utf-8
from utils import *
import torch
import os
import argparse
from transformers import GPT2LMHeadModel
import utils
from transformers.generation_utils import top_k_top_p_filtering


@torch.no_grad()
def generate(model, context, length, temperature=1.0, top_k=30, top_p=0.0,
             device='cpu'):
    inputs = torch.LongTensor(context).unsqueeze(0).to(device)

    out = model.generate(
        input_ids=inputs,
        max_length=length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=1.15,
        do_sample=True,
    )
    return out[0, :].tolist()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='生成设备')
    parser.add_argument('--length', default=-1, type=int, required=False, help='生成长度')
    parser.add_argument('--nsamples', default=10, type=int, required=False, help='生成几个样本')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成温度')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高几选一')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
                        help='模型参数')
    parser.add_argument('--vocab_file', default='data/vocab/vocab.txt', type=str, required=False, help='词表路径')
    parser.add_argument('--model_path', default='model/final_model', type=str, required=False, help='模型路径')
    parser.add_argument('--prefix', default='[CLS][MASK]', type=str, required=False, help='生成文章的开头')
    parser.add_argument('--suffix', default='[SEP]', type=str, required=False, help='生成文章的结尾')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False)

    return parser.parse_args()


def prepare_inputs(prefix, suffix, length, tokenizer):
    """
    前缀，后缀，中间长度
    """
    begin_token_id = tokenizer.convert_tokens_to_ids('<begin>')

    prefix_tokens = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(prefix)
    )
    suffix_tokens = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(suffix)
    )

    prefix_positions = [i for i in range(len(prefix_tokens))]

    suffix_tokens = suffix_tokens + [begin_token_id]
    # prefix的长度 + 补全内容的长度length
    suffix_positions = [i + len(prefix_tokens) + length + 1 for i in range(len(suffix_tokens))]

    return suffix_tokens + prefix_tokens, suffix_positions + prefix_positions, len(prefix_tokens)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    device = utils.get_device()
    tokenizer = get_tokenizer(args.vocab_file)

    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    while True:
        prefix = input("\n\nInput Text >> ")
        prefix_tokens = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(prefix)
        )
        for c in range(args.nsamples):
            out = generate(
                model=model,
                context=prefix_tokens,
                length=args.length + 10,
                temperature=args.temperature,
                top_k=args.topk,
                top_p=args.topp,
                device=device
            )
            print("=" * 40 + "=" * 40 + "\n")
            text = tokenizer.decode(out, clean_up_tokenization_spaces=True).replace(' ', '')
            print(text)


if __name__ == '__main__':
    main()
