from functools import partial

import torch
from datasets import Dataset
from peft.tuners.lora import LoraLayer
from transformers import Trainer, TrainingArguments, default_data_collator
from transformers.trainer_callback import PrinterCallback

from autotrain import logger
from autotrain.trainers.clm import utils
from autotrain.trainers.clm.params import LLMTrainingParams


def process_data(data, tokenizer, config):
    data = data.to_pandas()
    data = data.fillna("")

    data = data[[config.text_column]]
    if config.add_eos_token:
        data[config.text_column] = data[config.text_column] + tokenizer.eos_token
    data = Dataset.from_pandas(data)
    return data


def train(config):
    logger.info("Starting default/generic CLM training...")
    if isinstance(config, dict):
        config = LLMTrainingParams(**config)
    train_data, valid_data = utils.process_input_data(config)
    tokenizer = utils.get_tokenizer(config)
    train_data, valid_data = utils.process_data_with_chat_template(config, tokenizer, train_data, valid_data)

    train_data = process_data(
        data=train_data,
        tokenizer=tokenizer,
        config=config,
    )
    if config.valid_split is not None:
        valid_data = process_data(
            data=valid_data,
            tokenizer=tokenizer,
            config=config,
        )

    logging_steps = utils.configure_logging_steps(config, train_data, valid_data)
    training_args = utils.configure_training_args(config, logging_steps)
    config = utils.configure_block_size(config, tokenizer)
    args = TrainingArguments(**training_args)

    model = utils.get_model(config, tokenizer)

    tokenize_fn = partial(utils.tokenize, tokenizer=tokenizer, config=config)
    group_texts_fn = partial(utils.group_texts, config=config)

    train_data = train_data.map(
        tokenize_fn,
        batched=True,
        num_proc=1,
        remove_columns=list(train_data.features),
        desc="Running tokenizer on train dataset",
    )

    if config.valid_split is not None:
        valid_data = valid_data.map(
            tokenize_fn,
            batched=True,
            num_proc=1,
            remove_columns=list(valid_data.features),
            desc="Running tokenizer on validation dataset",
        )

    train_data = train_data.map(
        group_texts_fn,
        batched=True,
        num_proc=4,
        desc=f"Grouping texts in chunks of {config.block_size}",
    )

    if config.valid_split is not None:
        valid_data = valid_data.map(
            group_texts_fn,
            batched=True,
            num_proc=4,
            desc=f"Grouping texts in chunks of {config.block_size}",
        )

    logger.info("creating trainer")
    callbacks = utils.get_callbacks(config)
    trainer_args = dict(
        args=args,
        model=model,
        callbacks=callbacks,
    )
    trainer = Trainer(
        **trainer_args,
        train_dataset=train_data,
        eval_dataset=valid_data if config.valid_split is not None else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    for name, module in trainer.model.named_modules():
        if isinstance(module, LoraLayer):
            if config.mixed_precision == "bf16":
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if config.mixed_precision == "bf16" and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    trainer.remove_callback(PrinterCallback)
    trainer.train()
    utils.post_training_steps(config, trainer)
