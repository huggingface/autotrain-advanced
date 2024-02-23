import argparse
import json
import os
import sys
from enum import Enum
from functools import partial

import torch
from accelerate.state import PartialState
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from trl import DPOTrainer, RewardConfig, RewardTrainer, SFTTrainer

from autotrain import logger
from autotrain.trainers.clm import utils
from autotrain.trainers.clm.callbacks import LoadBestPeftModelCallback, SavePeftModelCallback
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.common import monitor, pause_space, remove_autotrain_data, save_training_params


class ZephyrSpecialTokens(str, Enum):
    USER = "<|user|>"
    ASSISTANT = "<|assistant|>"
    SYSTEM = "<|system|>"
    EOS_TOKEN = "</s>"
    BOS_TOKEN = "<s>"
    PAD_TOKEN = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class ChatmlSpecialTokens(str, Enum):
    USER = "<|im_start|>user"
    ASSISTANT = "<|im_start|>assistant"
    SYSTEM = "<|im_start|>system"
    EOS_TOKEN = "<|im_end|>"
    BOS_TOKEN = "<s>"
    PAD_TOKEN = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


def parse_args():
    # get training_config.json from the end user
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()


def process_input_data(config):
    if config.data_path == f"{config.project_name}/autotrain-data":
        logger.info("loading dataset from disk")
        train_data = load_from_disk(config.data_path)[config.train_split]
    else:
        train_data = load_dataset(
            config.data_path,
            split=config.train_split,
            token=config.token,
        )
    # rename columns for reward trainer
    if config.trainer in ("dpo", "reward"):
        if not (config.text_column == "chosen" and config.text_column in train_data.column_names):
            train_data = train_data.rename_column(config.text_column, "chosen")
        if not (config.rejected_text_column == "rejected" and config.rejected_text_column in train_data.column_names):
            train_data = train_data.rename_column(config.rejected_text_column, "rejected")
    if config.trainer == "dpo":
        if not (config.prompt_text_column == "prompt" and config.prompt_text_column in train_data.column_names):
            train_data = train_data.rename_column(config.prompt_text_column, "prompt")

    if config.valid_split is not None:
        if config.data_path == f"{config.project_name}/autotrain-data":
            valid_data = load_from_disk(config.data_path)[config.valid_split]
        else:
            valid_data = load_dataset(
                config.data_path,
                split=config.valid_split,
                token=config.token,
            )

        if config.trainer in ("dpo", "reward"):
            if not (config.text_column == "chosen" and config.text_column in valid_data.column_names):
                valid_data = valid_data.rename_column(config.text_column, "chosen")
            if not (
                config.rejected_text_column == "rejected" and config.rejected_text_column in valid_data.column_names
            ):
                valid_data = valid_data.rename_column(config.rejected_text_column, "rejected")
        if config.trainer == "dpo":
            if not (config.prompt_text_column == "prompt" and config.prompt_text_column in valid_data.column_names):
                valid_data = valid_data.rename_column(config.prompt_text_column, "prompt")
    else:
        valid_data = None

    logger.info(f"Train data: {train_data}")
    logger.info(f"Valid data: {valid_data}")

    return train_data, valid_data


@monitor
def train(config):
    if isinstance(config, dict):
        config = LLMTrainingParams(**config)

    if config.padding not in ("left", "right"):
        config.padding = None

    is_deepspeed_enabled = os.environ.get("ACCELERATE_USE_DEEPSPEED", "False").lower() == "true"

    if config.repo_id is None and config.username is not None:
        config.repo_id = f"{config.username}/{config.project_name}"

    train_data, valid_data = process_input_data(config)

    special_tokens = None
    chat_template = None
    if config.chat_template == "chatml":
        special_tokens = ChatmlSpecialTokens
        chat_template = utils.CHATML_CHAT_TEMPLATE
    elif config.chat_template == "zephyr":
        special_tokens = ZephyrSpecialTokens
        chat_template = utils.ZEPHYR_CHAT_TEMPLATE

    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            pad_token=special_tokens.PAD_TOKEN.value,
            bos_token=special_tokens.BOS_TOKEN.value,
            eos_token=special_tokens.EOS_TOKEN.value,
            additional_special_tokens=special_tokens.list(),
            token=config.token,
            trust_remote_code=True,
        )
        tokenizer.chat_template = chat_template
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model, token=config.token, trust_remote_code=True)
        if tokenizer.chat_template is None:
            tokenizer.chat_template = utils.DEFAULT_CHAT_TEMPLATE

    if config.chat_template in ("chatml", "zephyr", "tokenizer"):
        train_data = train_data.map(
            utils.apply_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "config": config,
            },
        )
        if config.valid_split is not None:
            valid_data = valid_data.map(
                utils.apply_chat_template,
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "config": config,
                },
            )

    if tokenizer.model_max_length > 2048:
        tokenizer.model_max_length = config.model_max_length

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if config.padding:
        tokenizer.padding_side = config.padding

    if config.trainer == "default":
        train_data = utils.process_data(
            data=train_data,
            tokenizer=tokenizer,
            config=config,
        )
        if config.valid_split is not None:
            valid_data = utils.process_data(
                data=valid_data,
                tokenizer=tokenizer,
                config=config,
            )

    model_config = AutoConfig.from_pretrained(
        config.model,
        token=config.token,
        trust_remote_code=True,
    )
    if config.trainer == "reward":
        model_config.num_labels = 1
        model_config.pad_token_id = tokenizer.pad_token_id
        model_config.pad_token = tokenizer.pad_token

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

        if config.trainer == "reward":
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model,
                config=model_config,
                token=config.token,
                quantization_config=bnb_config,
                trust_remote_code=True,
                use_flash_attention_2=config.use_flash_attention_2,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.model,
                config=model_config,
                token=config.token,
                quantization_config=bnb_config,
                trust_remote_code=True,
                use_flash_attention_2=config.use_flash_attention_2,
            )
            model_ref = None
    else:
        if config.trainer == "reward":
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model,
                trust_remote_code=True,
                num_labels=1,
                use_flash_attention_2=config.use_flash_attention_2,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.model,
                config=model_config,
                token=config.token,
                trust_remote_code=True,
                use_flash_attention_2=config.use_flash_attention_2,
            )
            if config.model_ref is not None:
                model_ref = AutoModelForCausalLM.from_pretrained(
                    config.model_ref,
                    config=model_config,
                    token=config.token,
                    trust_remote_code=True,
                    use_flash_attention_2=config.use_flash_attention_2,
                )
            else:
                model_ref = None

    model.resize_token_embeddings(len(tokenizer))
    if model_ref is not None:
        model_ref.resize_token_embeddings(len(tokenizer))

    if config.peft:
        if config.quantization is not None:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=not config.disable_gradient_checkpointing,
            )
        else:
            model.enable_input_require_grads()

        if config.trainer == "reward":
            peft_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="SEQ_CLS",
                target_modules=utils.get_target_modules(config),
                # modules_to_save=["scores"],
            )
        else:
            peft_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=utils.get_target_modules(config),
            )
        model = get_peft_model(model, peft_config)

    if config.block_size == -1:
        config.block_size = None

    if config.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if config.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({config.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(config.block_size, tokenizer.model_max_length)

    config.block_size = block_size
    logger.info(f"Using block size {block_size}")

    if config.trainer == "default":
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
            desc=f"Grouping texts in chunks of {block_size}",
        )

        if config.valid_split is not None:
            valid_data = valid_data.map(
                group_texts_fn,
                batched=True,
                num_proc=4,
                desc=f"Grouping texts in chunks of {block_size}",
            )

    elif config.trainer == "reward":
        reward_proc = partial(utils.preprocess_reward, tokenizer=tokenizer)
        train_data = train_data.map(
            reward_proc,
            batched=True,
            num_proc=4,
            desc="Running tokenizer on train dataset",
        )
        train_data = train_data.filter(
            lambda x: len(x["input_ids_chosen"]) <= config.block_size
            and len(x["input_ids_rejected"]) <= config.block_size
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
    # trainer specific
    if config.logging_steps == -1:
        if config.valid_split is not None:
            logging_steps = int(0.2 * len(valid_data) / config.batch_size)
        else:
            logging_steps = int(0.2 * len(train_data) / config.batch_size)
        if logging_steps == 0:
            logging_steps = 1

    else:
        logging_steps = config.logging_steps

    training_args = dict(
        output_dir=config.project_name,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        evaluation_strategy=config.evaluation_strategy if config.valid_split is not None else "no",
        logging_steps=logging_steps,
        save_total_limit=config.save_total_limit,
        save_strategy=config.save_strategy,
        gradient_accumulation_steps=config.gradient_accumulation,
        report_to=config.log,
        auto_find_batch_size=config.auto_find_batch_size,
        lr_scheduler_type=config.scheduler,
        optim=config.optimizer,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        push_to_hub=False,
        load_best_model_at_end=True if config.valid_split is not None else False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=not config.disable_gradient_checkpointing,
        remove_unused_columns=False,
    )

    if config.mixed_precision == "fp16":
        training_args["fp16"] = True
    if config.mixed_precision == "bf16":
        training_args["bf16"] = True

    if config.trainer == "reward":
        training_args["max_length"] = config.block_size
        args = RewardConfig(**training_args)
    else:
        args = TrainingArguments(**training_args)

    callbacks = []
    if config.peft and not is_deepspeed_enabled:
        callbacks.append(SavePeftModelCallback)
        if config.valid_split is not None:
            callbacks.append(LoadBestPeftModelCallback)

    # if config.peft and is_deepspeed_enabled:
    #     callbacks.append(SaveDeepSpeedPeftModelCallback)

    trainer_args = dict(
        args=args,
        model=model,
    )

    if config.trainer == "default":
        trainer = Trainer(
            **trainer_args,
            train_dataset=train_data,
            eval_dataset=valid_data if config.valid_split is not None else None,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            callbacks=callbacks,
        )
    elif config.trainer == "sft":
        trainer = SFTTrainer(
            **trainer_args,
            train_dataset=train_data,
            eval_dataset=valid_data if config.valid_split is not None else None,
            peft_config=peft_config if config.peft else None,
            dataset_text_field=config.text_column,
            max_seq_length=config.block_size,
            tokenizer=tokenizer,
            packing=True,
        )
    elif config.trainer == "reward":
        trainer = RewardTrainer(
            **trainer_args,
            train_dataset=train_data,
            eval_dataset=valid_data if config.valid_split is not None else None,
            peft_config=peft_config if config.peft else None,
            tokenizer=tokenizer,
        )
    elif config.trainer == "dpo":
        if isinstance(config.block_size, int):
            max_length = config.block_size
            max_prompt_length = None
            max_target_length = None
        elif isinstance(config.block_size, list):
            if len(config.block_size) == 3:
                max_length, max_prompt_length, max_target_length = config.block_size
            elif len(config.block_size) == 2:
                max_length, max_prompt_length = config.block_size
                max_target_length = None
            else:
                raise ValueError(f"block_size must be a list of length 2 or 3, got {config.block_size}")
        else:
            raise ValueError(f"block_size must be an int or a list, got {config.block_size}")
        trainer = DPOTrainer(
            **trainer_args,
            ref_model=model_ref,
            beta=config.dpo_beta,
            train_dataset=train_data,
            eval_dataset=valid_data if config.valid_split is not None else None,
            tokenizer=tokenizer,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            max_target_length=max_target_length,
            peft_config=peft_config if config.peft else None,
        )
    else:
        raise ValueError(f"trainer `{config.trainer}` not supported")
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

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

    trainer.train()

    logger.info("Finished training, saving model...")
    trainer.save_model(config.project_name)

    model_card = utils.create_model_card()

    # save model card to output directory as README.md
    with open(f"{config.project_name}/README.md", "w") as f:
        f.write(model_card)

    if config.peft:
        if config.merge_adapter:
            logger.info("Merging adapter weights...")
            try:
                utils.merge_adapter(
                    base_model_path=config.model,
                    target_model_path=config.project_name,
                    adapter_path=config.project_name,
                )
                # remove adapter weights: adapter_*
                for file in os.listdir(config.project_name):
                    if file.startswith("adapter_"):
                        os.remove(f"{config.project_name}/{file}")
            except Exception as e:
                logger.warning(f"Failed to merge adapter weights: {e}")
                logger.warning("Skipping adapter merge. Only adapter weights will be saved.")
        else:
            utils.create_peft_handler(config)
            utils.create_requirements_txt(config)

    if config.push_to_hub:
        if PartialState().process_index == 0:
            # remove data folder
            remove_autotrain_data(config)
            logger.info("Pushing model to hub...")
            save_training_params(config)
            api = HfApi(token=config.token)
            api.create_repo(repo_id=config.repo_id, repo_type="model", private=True)
            api.upload_folder(
                folder_path=config.project_name,
                repo_id=config.repo_id,
                repo_type="model",
            )

    if PartialState().process_index == 0:
        pause_space(config)


if __name__ == "__main__":
    _args = parse_args()
    training_config = json.load(open(_args.training_config))
    _config = LLMTrainingParams(**training_config)
    train(_config)
