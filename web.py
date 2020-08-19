#!/usr/bin/env python
# coding=utf-8
from utils import *
import os
import argparse
from transformers import GPT2LMHeadModel
import utils
from flask import request, Flask
from random import randint
import json

model = None
tokenizer = None
app = Flask(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8080, type=int, required=False, help='监听端口')
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='生成设备')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
                        help='模型参数')
    parser.add_argument('--vocab_file', default='data/vocab/vocab.txt', type=str, required=False, help='词表路径')
    parser.add_argument('--model_path', default='model/final_model', type=str, required=False, help='模型路径')

    return parser.parse_args()


def load_model():
    """Load the pre-trained model, you can use your model just as easily.
    """
    global model, tokenizer, args
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    device = utils.get_device()
    tokenizer = get_tokenizer(args.vocab_file)

    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()


@torch.no_grad()
def generate(model, context, length, temperature=1.0, top_k=30, top_p=0.0, repetition_penalty=1.0,
             device='cpu'):
    torch.manual_seed(randint(999, 9999999))

    inputs = torch.LongTensor(context).unsqueeze(0).to(device)

    out = model.generate(
        input_ids=inputs,
        max_length=length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
    )
    return out[0, :].tolist()


@app.route("/predict", methods=["GET", "POST"])
def predict():
    global tokenizer, args
    prefix = request.args["prefix"]
    topp = float(request.args['topp'])
    topk = int(request.args['topk'])
    maxlen = int(request.args['len'])
    repetition_penalty = float(request.args['repetition_penalty'])
    temperature = float(request.args['temperature'])

    cb = request.args["callback"]
    data = {"code": 0}

    try:
        if len(prefix) <= 0:
            return cb + "(" + json.dumps(data) + ")"
        prefix_tokens = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(prefix)
        )

        out = generate(
            model=model,
            context=prefix_tokens,
            length=maxlen,
            temperature=temperature,
            top_k=topk,
            top_p=topp,
            repetition_penalty=repetition_penalty,
        )
        text = tokenizer.decode(out, clean_up_tokenization_spaces=True).replace(' ', '')
        data["body"] = text

    except ValueError:
        pass
    except RuntimeError:
        pass

    return cb + "(" + json.dumps(data) + ")"


def main():
    global app, args
    args = parse_args()
    load_model()
    app.run(port=args.port)


if __name__ == '__main__':
    main()
