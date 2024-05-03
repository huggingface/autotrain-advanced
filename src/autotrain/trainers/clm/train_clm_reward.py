from functools import partial

import torch
from peft import LoraConfig
from transformers import AutoConfig, AutoModelForSequenceClassification, BitsAndBytesConfig
from transformers.trainer_callback import PrinterCallback
from trl import RewardConfig, RewardTrainer

from autotrain import logger
from autotrain.trainers.clm import utils
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.common import ALLOW_REMOTE_CODE


def train(config):
    logger.info("Starting Reward training...")
    if isinstance(config, dict):
        config = LLMTrainingParams(**config)
    train_data, valid_data = utils.process_input_data(config)
    tokenizer = utils.get_tokenizer(config)
    train_data, valid_data = utils.process_data_with_chat_template(config, tokenizer, train_data, valid_data)

    logging_steps = utils.configure_logging_steps(config, train_data, valid_data)
    training_args = utils.configure_training_args(config, logging_steps)
    config = utils.configure_block_size(config, tokenizer)
    training_args["max_length"] = config.block_size
    args = RewardConfig(**training_args)

    logger.info("loading model config...")
    model_config = AutoConfig.from_pretrained(
        config.model,
        token=config.token,
        trust_remote_code=ALLOW_REMOTE_CODE,
        use_cache=config.disable_gradient_checkpointing,
    )

    model_config.num_labels = 1
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config.pad_token = tokenizer.pad_token

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

        model = AutoModelForSequenceClassification.from_pretrained(
            config.model,
            config=model_config,
            token=config.token,
            quantization_config=bnb_config,
            trust_remote_code=ALLOW_REMOTE_CODE,
            use_flash_attention_2=config.use_flash_attention_2,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model,
            config=model_config,
            token=config.token,
            trust_remote_code=ALLOW_REMOTE_CODE,
            use_flash_attention_2=config.use_flash_attention_2,
        )

    logger.info(f"model dtype: {model.dtype}")
    model.resize_token_embeddings(len(tokenizer))

    if config.peft:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=utils.get_target_modules(config),
        )

    reward_proc = partial(utils.preprocess_reward, tokenizer=tokenizer)
    train_data = train_data.map(
        reward_proc,
        batched=True,
        num_proc=4,
        desc="Running tokenizer on train dataset",
    )
    train_data = train_data.filter(
        lambda x: len(x["input_ids_chosen"]) <= config.block_size and len(x["input_ids_rejected"]) <= config.block_size
    )
    if config.valid_split is not None:
        valid_data = valid_data.map(
            reward_proc,
            batched=True,
            num_proc=4,
            desc="Running tokenizer on validation dataset",
        )
        valid_data = valid_data.filter(
            lambda x: len(x["input_ids_chosen"]) <= config.block_size
            and len(x["input_ids_rejected"]) <= config.block_size
        )

    logger.info("creating trainer")
    callbacks = utils.get_callbacks(config)
    trainer_args = dict(
        args=args,
        model=model,
        callbacks=callbacks,
    )
    trainer = RewardTrainer(
        **trainer_args,
        train_dataset=train_data,
        eval_dataset=valid_data if config.valid_split is not None else None,
        peft_config=peft_config if config.peft else None,
        tokenizer=tokenizer,
    )

    trainer.remove_callback(PrinterCallback)
    trainer.train()
    utils.post_training_steps(config, trainer)
