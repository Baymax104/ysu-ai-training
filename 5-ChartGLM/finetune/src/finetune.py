# -*- coding: utf-8 -*-
import functools
import os
from typing import Annotated, Union

import typer
from datasets import Split
from peft import PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from config import FinetuningConfig
from dataset import (
    process_batch,
    process_batch_eval,
    DataManager,
    GLMDataCollator
)
from trainer import (
    load_tokenizer_and_model,
    prepare_model_for_training,
    GLMTrainer,
    compute_metrics
)

ModelType = Union[PreTrainedModel, PeftModelForCausalLM, AutoModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer]
app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
        data_dir: Annotated[str, typer.Argument()],
        model_dir: Annotated[str, typer.Argument()],
        config_file: Annotated[str, typer.Argument()],
        auto_resume_from_checkpoint: str = typer.Argument(default='')
):
    ft_config = FinetuningConfig.from_file(config_file)
    tokenizer, model = load_tokenizer_and_model(model_dir, peft_config=ft_config.peft_config)
    data_manager = DataManager(data_dir, ft_config.data_config)

    train_dataset = data_manager.get_dataset(
        Split.TRAIN,
        functools.partial(
            process_batch,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    print('train_dataset:', train_dataset)
    val_dataset = data_manager.get_dataset(
        Split.VALIDATION,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if val_dataset is not None:
        print('val_dataset:', val_dataset)
    test_dataset = data_manager.get_dataset(
        Split.TEST,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if test_dataset is not None:
        print('test_dataset:', test_dataset)

    # checks encoded dataset
    # utils.sanity_check(train_dataset[0]["input_ids"], train_dataset[0]["labels"], tokenizer)

    # turn model to fp32
    prepare_model_for_training(model, ft_config.training_args.use_cpu)

    ft_config.training_args.generation_config.pad_token_id = tokenizer.pad_token_id
    ft_config.training_args.generation_config.eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.get_command('<|user|>'),
        tokenizer.get_command('<|observation|>'),
    ]
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    use_tokenizer = True
    if ft_config.peft_config is not None:
        use_tokenizer = ft_config.peft_config.peft_type != "LORA"

    trainer = GLMTrainer(
        model=model,
        args=ft_config.training_args,
        data_collator=GLMDataCollator(
            tokenizer=tokenizer,
            padding='longest',
            return_tensors='pt',
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset.select(list(range(50))),
        tokenizer=tokenizer if use_tokenizer else None,  # LORA does not need tokenizer
        compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),
    )

    if auto_resume_from_checkpoint.upper() == "" or auto_resume_from_checkpoint is None:
        trainer.train()
    else:
        def do_rf_checkpoint(sn):
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            checkpoint_directory = os.path.join(output_dir, "checkpoint-" + sn)
            print("resume checkpoint from  checkpoint-" + sn)
            trainer.train(resume_from_checkpoint=checkpoint_directory)

        output_dir = ft_config.training_args.output_dir

        # resume from latest checkpoint
        if auto_resume_from_checkpoint.upper() == "YES":
            dirlist = os.listdir(output_dir)
            checkpoint_sn = 0
            # get latest checkpoint
            for checkpoint_str in dirlist:
                if checkpoint_str.find("eckpoint") > 0 and checkpoint_str.find("tmp") == -1:
                    checkpoint = int(checkpoint_str.replace("checkpoint-", ""))
                    if checkpoint > checkpoint_sn:
                        checkpoint_sn = checkpoint
            if checkpoint_sn > 0:
                do_rf_checkpoint(str(checkpoint_sn))
            else:
                trainer.train()
        else:
            # resume from specific checkpoint
            if auto_resume_from_checkpoint.isdigit() and int(auto_resume_from_checkpoint) > 0:
                do_rf_checkpoint(auto_resume_from_checkpoint)
            else:
                print(auto_resume_from_checkpoint,
                      f"The specified checkpoint sn({auto_resume_from_checkpoint}) has not been saved. Please search for the correct chkeckpoint in the model output directory")

    # test stage
    if test_dataset is not None:
        trainer.predict(test_dataset)


if __name__ == '__main__':
    app()
