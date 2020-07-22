#!/usr/bin/env python
# coding=utf-8
from utils import *
import torch
import os
import argparse
from transformers import GPT2LMHeadModel
import utils


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


@torch.no_grad()
def generate(model, context, context_positions, length, prefix_len, temperature=1.0, top_k=30, top_p=0.0,
             device='cpu'):
    inputs = torch.LongTensor(context).unsqueeze(0).to(device)
    inputs_positions = torch.LongTensor(context_positions).unsqueeze(0).to(device)

    generated_tokens = [] + context
    generated_position = prefix_len
    past = None
    for _ in range(length):
        output, past = model(inputs, past=past, position_ids=inputs_positions)
        output = output[-1].squeeze(0) / temperature
        filtered_logits = top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
        next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
        generated_tokens.append(next_token.item())
        inputs = next_token.view(1, 1)
        generated_position += 1
        inputs_positions = torch.LongTensor([generated_position]).unsqueeze(0).to(device)

    return generated_tokens


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
    tokenizer = utils.BertTokenizer(
        vocab_file=args.vocab_file
    )
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    for c in range(args.nsamples):
        context_tokens, context_positions, prefix_len = prepare_inputs(
            prefix=args.prefix,
            suffix=args.suffix,
            length=args.length,
            tokenizer=tokenizer
        )
        out = generate(
            model=model,
            context=context_tokens,
            context_positions=context_positions,
            length=args.length + 10,
            prefix_len=prefix_len,
            temperature=args.temperature,
            top_k=args.topk,
            top_p=args.topp,
            device=device
        )

        print("=" * 40 + "=" * 40 + "\n")
        text = tokenizer.decode(out, clean_up_tokenization_spaces=True)
        print(text)


if __name__ == '__main__':
    main()