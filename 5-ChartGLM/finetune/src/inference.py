#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Annotated

import typer
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

import utils

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    model_dir: Annotated[str, typer.Argument()],
    prompt: Annotated[str, typer.Argument()]
):
    model_dir = utils.resolve_path(model_dir)
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, device_map='auto')
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, device_map='auto')
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

    response, _ = model.chat(tokenizer, prompt)
    print(response)


if __name__ == '__main__':
    app()
