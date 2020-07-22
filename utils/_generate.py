#!/usr/bin/env python
# coding=utf-8
from utils import *
import torch
import os
import argparse
from transformers import GPT2LMHeadModel
import utils


class HaimGPT2LMHeadModel(GPT2LMHeadModel):
    def prepare_inputs_for_generation(self, input_ids, past, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        position_ids = kwargs["position_ids"]
        position_ids = position_ids.expand(input_ids.shape[0], input_ids.shape[1])

        return {
            "input_ids": input_ids,
            "past": past,
            "use_cache": kwargs["use_cache"],
            "position_ids": position_ids,
        }


def generate(model, tokenizer, context, context_positions, device, args):
    inputs = torch.LongTensor(context).unsqueeze(0).to(device)
    inputs_positions = torch.LongTensor(context_positions).unsqueeze(0).to(device)

    length = args.length + len(inputs) + 10

    output_sequences = model.generate(
        input_ids=inputs,
        max_length=length,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        repetition_penalty=args.repetition_penalty,
        do_sample=True,
        num_return_sequences=args.nsamples,
        pad_token_id=0,
        position_ids=inputs_positions,
    )

    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        print(text)


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
    end_token_id = tokenizer.convert_tokens_to_ids('<end>')
    prefix_tokens = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(prefix)
    )
    suffix_tokens = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(suffix)
    )

    prefix_positions = [i for i in range(len(prefix_tokens))]
    suffix_tokens = suffix_tokens + [begin_token_id]
    suffix_positions = [i + len(prefix_tokens) + length + 1 for i in
                        range(len(suffix_tokens))]  # prefix的长度 + 补全内容的长度length

    context_tokens = suffix_tokens + prefix_tokens
    context_positions = suffix_positions + prefix_positions

    return context_tokens, context_positions


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    device = utils.get_device()
    tokenizer = utils.BertTokenizer(
        vocab_file=args.vocab_file
    )
    model = HaimGPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    context_tokens, context_positions = prepare_inputs(
        prefix=args.prefix,
        suffix=args.suffix,
        length=args.length,
        tokenizer=tokenizer
    )
    generate(
        model=model,
        context=context_tokens,
        tokenizer=tokenizer,
        context_positions=context_positions,
        args=args,
        device=device,
    )


if __name__ == '__main__':
    main()
