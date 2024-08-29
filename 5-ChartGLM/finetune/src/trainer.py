# -*- coding: UTF-8 -*-
from typing import Any

import jieba
import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from peft import get_peft_model
from rouge_chinese import Rouge
from torch import nn
from transformers import (
    EvalPrediction,
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    Seq2SeqTrainer
)


def compute_metrics(eval_preds: EvalPrediction, tokenizer: PreTrainedTokenizer):
    batched_pred_ids, batched_label_ids = eval_preds

    metrics_dct = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-4': []}
    for pred_ids, label_ids in zip(batched_pred_ids, batched_label_ids):
        # when predication is empty, avoid raising ValueError
        if len(pred_ids) <= 0 or len(label_ids) <= 0:
            continue
        pred_txt = tokenizer.decode(pred_ids).strip()
        label_txt = tokenizer.decode(label_ids).strip()
        pred_tokens = list(jieba.cut(pred_txt))
        label_tokens = list(jieba.cut(label_txt))
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
        for k, v in scores[0].items():
            metrics_dct[k].append(round(v['f'] * 100, 4))
        bleu = sentence_bleu([label_tokens], pred_tokens, smoothing_function=SmoothingFunction().method3)
        metrics_dct['bleu-4'].append(bleu)
    return {k: np.mean(v) for k, v in metrics_dct.items()}


def print_model_size(model: PreTrainedModel):
    print("--> Model")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> model has {total_params / 1e6}M params\n")


def prepare_model_for_training(model: nn.Module, use_cpu: bool):
    for param in model.parameters():
        if param.requires_grad or use_cpu:
            param.data = param.data.to(torch.float32)


def load_tokenizer_and_model(model_dir: str, peft_config=None):
    model = None
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if peft_config is not None:
        if peft_config.peft_type.name == "PREFIX_TUNING":
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            config.pre_seq_len = peft_config.num_virtual_tokens
            config.use_cache = False
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                config=config,
            )
        if peft_config.peft_type.name == "LORA":
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                empty_init=False,
                use_cache=False
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            empty_init=False,
            use_cache=False
        )
    print_model_size(model)
    return tokenizer, model


class GLMTrainer(Seq2SeqTrainer):

    def prediction_step(self,
                        model: nn.Module,
                        inputs: dict[str, Any],
                        prediction_loss_only: bool,
                        ignore_keys=None,
                        **gen_kwargs):
        output_ids = None
        if self.args.predict_with_generate:
            output_ids = inputs.pop('output_ids')
        input_ids = inputs['input_ids']
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
        )
        generated_tokens = generated_tokens[:, input_ids.size()[1]:]
        if self.args.predict_with_generate:
            labels = output_ids
        return loss, generated_tokens, labels

    # For P-Tuning a new save_model function is fine for the prefix_encoder model
    # but may cost problems for the whole model loading

    # def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
    #     if output_dir is None:
    #         output_dir = self.args.output_dir
    #     os.makedirs(output_dir, exist_ok=True)
    #     ptuning_params = {k: v for k, v in self.model.transformer.prefix_encoder.state_dict().items()}
    #
    #     torch.save(ptuning_params, os.path.join(output_dir, 'pytorch_model.bin'))
    #
    #     print(f"P-Tuning model weights saved in {output_dir}")
    #
    #     if self.tokenizer is not None:
    #         self.tokenizer.save_pretrained(output_dir)
