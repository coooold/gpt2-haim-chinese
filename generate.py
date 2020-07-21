#!/usr/bin/env python
# coding=utf-8
from utils import *
import torch
import os
import argparse
from transformers import GPT2LMHeadModel
import utils


def is_word(word):
    for item in list(word):
        if item not in 'qwertyuiopasdfghjklzxcvbnm':
            return False
    return True


def _is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def generate(model, prev, prev_position, context, context_positions, length, temperature=1.0, top_k=30, top_p=0.0,
             device='cpu'):
    prev = torch.LongTensor([prev]).unsqueeze(0).to(device)
    inputs = torch.LongTensor(context).unsqueeze(0).to(device)
    inputs_positions = torch.LongTensor(context_positions).unsqueeze(0).to(device)

    _, past = model(
        input_ids=inputs,
        past=None,
        position_ids=inputs_positions,
    )
    generate = [] + context + [prev]
    with torch.no_grad():
        for _ in range(length):
            prev_positions = torch.LongTensor([prev_position]).unsqueeze(0).to(device)
            output, past = model(prev, past=past, position_ids=prev_positions)
            output = output[-1].squeeze(0) / temperature
            filtered_logits = top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
            generate.append(next_token.item())
            prev = next_token.view(1, 1)
            prev_position += 1
    return generate


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

    prev = prefix_tokens[-1]

    suffix_tokens = suffix_tokens + [begin_token_id]
    suffix_positions = [i + len(prefix_tokens) + length for i in range(len(suffix_tokens))]  # prefix的长度 + 补全内容的长度length

    prefix_tokens = prefix_tokens[:-1]  # 拿走一个当成prev
    prefix_positions = [i for i in range(len(prefix_tokens))]

    context_tokens = suffix_tokens + prefix_tokens
    context_positions = suffix_positions + prefix_positions

    prev_position = len(prefix_tokens) + 1

    return prev, prev_position, context_tokens, context_positions


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    device = utils.get_device()
    tokenizer = utils.BertTokenizer(
        vocab_file=args.vocab_file
    )
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    for c in range(args.nsamples):
        prev, prev_position, context_tokens, context_positions = prepare_inputs(
            prefix=args.prefix,
            suffix=args.suffix,
            length=args.length,
            tokenizer=tokenizer
        )
        out = generate(
            model=model,
            prev=prev,
            prev_position=prev_position,
            context=context_tokens,
            context_positions=context_positions,
            length=args.length,
            temperature=args.temperature,
            top_k=args.topk,
            top_p=args.topp,
            device=device
        )

        text = tokenizer.convert_ids_to_tokens(out)
        for i, item in enumerate(text[:-1]):  # 确保英文前后有空格
            if is_word(item) and is_word(text[i + 1]):
                text[i] = item + ' '
        for i, item in enumerate(text):
            if item == '[MASK]':
                text[i] = ''
            elif item == '[CLS]':
                text[i] = '\n\n'
            elif item == '[SEP]':
                text[i] = '\n'
        print("=" * 40 + "=" * 40 + "\n")
        text = ''.join(text).replace('##', '').strip()
        print(text)


if __name__ == '__main__':
    main()
    exit(0)
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    device = utils.get_device()

    tokenizer = utils.BertTokenizer(
        vocab_file=args.vocab_file
    )
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    n_ctx = model.config.n_ctx

    while True:
        context_tokens = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(args.prefix)
        )
        generated = 0
        for _ in range(args.nsamples // args.batch_size):
            gen_len = args.length + 10  # 多生成点，可能能发现<end>
            out = generate(
                model=model,
                context=context_tokens,
                length=gen_len,
                temperature=args.temperature,
                top_k=args.topk,
                top_p=args.topp,
                device=device
            )
            for i in range(args.batch_size):
                generated += 1
                text = tokenizer.convert_ids_to_tokens(out)
                for i, item in enumerate(text[:-1]):  # 确保英文前后有空格
                    if is_word(item) and is_word(text[i + 1]):
                        text[i] = item + ' '
                    if item == '<end>':
                        text[i] = item + ' '
                        break
                for i, item in enumerate(text):
                    if item == '[MASK]':
                        text[i] = ''
                    elif item == '[CLS]':
                        text[i] = '\n\n'
                    elif item == '[SEP]':
                        text[i] = '\n'
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + "\n")
                text = ''.join(text).replace('##', '').strip()
                print(text)
        if generated == args.nsamples:
            break
