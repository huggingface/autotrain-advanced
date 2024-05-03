from functools import partial

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_callback import PrinterCallback

from autotrain import logger
from autotrain.trainers.clm import utils
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.common import ALLOW_REMOTE_CODE


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

    logger.info("loading model config...")
    model_config = AutoConfig.from_pretrained(
        config.model,
        token=config.token,
        trust_remote_code=ALLOW_REMOTE_CODE,
        use_cache=config.disable_gradient_checkpointing,
    )

    logger.info("loading model...")
    if config.peft:
        if config.quantization == "int4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
        elif config.quantization == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_config = None

        model = AutoModelForCausalLM.from_pretrained(
            config.model,
            config=model_config,
            token=config.token,
            quantization_config=bnb_config,
            trust_remote_code=ALLOW_REMOTE_CODE,
            use_flash_attention_2=config.use_flash_attention_2,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model,
            config=model_config,
            token=config.token,
            trust_remote_code=ALLOW_REMOTE_CODE,
            use_flash_attention_2=config.use_flash_attention_2,
        )

    logger.info(f"model dtype: {model.dtype}")
    model.resize_token_embeddings(len(tokenizer))
    if config.peft:
        logger.info("preparing peft model...")
        if config.quantization is not None:
            gradient_checkpointing_kwargs = {}
            if not config.disable_gradient_checkpointing:
                if config.quantization in ("int4", "int8"):
                    gradient_checkpointing_kwargs = {"use_reentrant": True}
                else:
                    gradient_checkpointing_kwargs = {"use_reentrant": False}
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=not config.disable_gradient_checkpointing,
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            )
        else:
            model.enable_input_require_grads()

        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=utils.get_target_modules(config),
        )
        model = get_peft_model(model, peft_config)

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
