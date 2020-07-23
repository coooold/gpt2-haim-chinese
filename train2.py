#!/usr/bin/env python
# coding=utf-8
from utils import *
from transformers import Trainer
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from torch import nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class HaimTrainer(Trainer):
    def _training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        print(loss)
        exit(0)

        loss.backward()

        return loss.item()

    def __training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], optimizer: torch.optim.Optimizer
    ) -> float:

        suffix_lens = inputs['suffix_len']
        batch_size = suffix_lens.shape[0]

        input_ids = inputs['input_ids']
        position_ids = inputs['position_ids']
        labels = inputs['labels']

        suffix_ids = []
        suffix_position_ids = []
        suffix_labels = []
        prefix_ids = []
        prefix_position_ids = []
        prefix_labels = []
        for i in range(batch_size):
            suffix_len = suffix_lens[i]
            suffix_ids.append(input_ids[i][:suffix_len].tolist())
            suffix_position_ids.append(position_ids[i][:suffix_len])
            suffix_labels.append(labels[i][:suffix_len])
            prefix_ids.append(input_ids[i][suffix_len:])
            prefix_position_ids.append(position_ids[i][suffix_len:])
            prefix_labels.append(labels[i][suffix_len:])

        suffix_ids = torch.LongTensor(suffix_ids).to(self.args.device)
        suffix_position_ids = torch.LongTensor(suffix_position_ids).to(self.args.device)
        suffix_labels = torch.LongTensor(suffix_labels).to(self.args.device)
        prefix_ids = torch.LongTensor(prefix_ids).to(self.args.device)
        prefix_position_ids = torch.LongTensor(prefix_position_ids).to(self.args.device)
        prefix_labels = torch.LongTensor(prefix_labels).to(self.args.device)

        model.train()

        # 输入suffix，suffix不计算loss
        _, past = model(
            input_ids=suffix_ids,
            position_ids=suffix_position_ids,
            labels=suffix_labels,
            past=None,
        )[:2]

        outputs = model(
            input_ids=prefix_ids,
            position_ids=prefix_position_ids,
            labels=prefix_labels,
            past=past
        )
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss.item()


def main():
    model_args, training_args = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # 创建模型
    model_config = GPT2Config.from_json_file(model_args.model_config_file)
    if not model_args.pretrained_model_path:
        model = GPT2LMHeadModel(config=model_config)
    else:
        model = GPT2LMHeadModel.from_pretrained(model_args.pretrained_model_path)

    # 计算参数数量
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += parameter.numel()
    logger.info('number of parameters: {}'.format(num_parameters))

    full_tokenizer = BertTokenizer(
        vocab_file=model_args.vocab_file
    )

    # 输入集
    train_dataset = GPT2Dataset(model_config.n_ctx,
                                stride=model_args.stride,
                                tokenized_file_path=model_args.data_dir,
                                tokenizer=full_tokenizer)

    trainer = HaimTrainer(model=model,
                          args=training_args,
                          train_dataset=train_dataset)

    # 开始训练
    trainer.train(
        model_path=model_args.pretrained_model_path
    )
    trainer.save_model()


if __name__ == "__main__":
    main()
