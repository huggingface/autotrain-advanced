import argparse
import json
import os
import sys
from functools import partial

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.state import PartialState
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
from trl import RewardConfig, RewardTrainer, SFTTrainer

from autotrain import logger
from autotrain.trainers.clm import utils
from autotrain.trainers.clm.callbacks import LoadBestPeftModelCallback, SavePeftModelCallback
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.utils import monitor


def parse_args():
    # get training_config.json from the end user
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()


@monitor
def train(config):
    if isinstance(config, dict):
        config = LLMTrainingParams(**config)

    if config.repo_id is None and config.username is not None:
        config.repo_id = f"{config.username}/{config.project_name}"

    # check if config.train_split.csv exists in config.data_path
    if config.train_split is not None:
        train_path = f"{config.data_path}/{config.train_split}.csv"
        if os.path.exists(train_path):
            logger.info("loading dataset from csv")
            train_data = pd.read_csv(train_path)
            train_data = Dataset.from_pandas(train_data)
        else:
            train_data = load_dataset(
                config.data_path,
                split=config.train_split,
                token=config.token,
            )
        # rename columns for reward trainer
        if config.trainer == "reward":
            if not (config.text_column == "chosen" and config.text_column in train_data.column_names):
                train_data = train_data.rename_column(config.text_column, "chosen")
            if not (
                config.rejected_text_column == "rejected" and config.rejected_text_column in train_data.column_names
            ):
                train_data = train_data.rename_column(config.rejected_text_column, "rejected")

    if config.valid_split is not None:
        valid_path = f"{config.data_path}/{config.valid_split}.csv"
        if os.path.exists(valid_path):
            logger.info("loading dataset from csv")
            valid_data = pd.read_csv(valid_path)
            valid_data = Dataset.from_pandas(valid_data)
        else:
            valid_data = load_dataset(
                config.data_path,
                split=config.valid_split,
                token=config.token,
            )

        if config.trainer == "reward":
            if not (config.text_column == "chosen" and config.text_column in valid_data.column_names):
                valid_data = valid_data.rename_column(config.text_column, "chosen")
            if not (
                config.rejected_text_column == "rejected" and config.rejected_text_column in valid_data.column_names
            ):
                valid_data = valid_data.rename_column(config.rejected_text_column, "rejected")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        token=config.token,
        trust_remote_code=True,
    )

    if tokenizer.model_max_length > 2048:
        tokenizer.model_max_length = config.model_max_length

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

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

    if config.use_peft:
        if config.use_int4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=config.use_int4,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
            config.fp16 = True
        elif config.use_int8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=config.use_int8)
            config.fp16 = True
        else:
            bnb_config = None

        if config.trainer == "reward":
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model,
                config=model_config,
                token=config.token,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map={"": Accelerator().process_index} if torch.cuda.is_available() else None,
                trust_remote_code=True,
                use_flash_attention_2=config.use_flash_attention_2,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.model,
                config=model_config,
                token=config.token,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map={"": Accelerator().process_index} if torch.cuda.is_available() else None,
                trust_remote_code=True,
                use_flash_attention_2=config.use_flash_attention_2,
            )
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

    model.resize_token_embeddings(len(tokenizer))

    if config.use_peft:
        if config.use_int8 or config.use_int4:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=not config.disable_gradient_checkpointing,
            )
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
        report_to="tensorboard",
        auto_find_batch_size=config.auto_find_batch_size,
        lr_scheduler_type=config.scheduler,
        optim=config.optimizer,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16,
        push_to_hub=False,
        load_best_model_at_end=True if config.valid_split is not None else False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=not config.disable_gradient_checkpointing,
    )

    if config.trainer == "reward":
        training_args["max_length"] = config.block_size
        args = RewardConfig(**training_args)
    else:
        args = TrainingArguments(**training_args)

    callbacks = []
    if config.use_peft:
        callbacks.append(SavePeftModelCallback)
        if config.valid_split is not None:
            callbacks.append(LoadBestPeftModelCallback)

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
            peft_config=peft_config if config.use_peft else None,
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
            peft_config=peft_config,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"trainer `{config.trainer}` not supported")
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    for name, module in trainer.model.named_modules():
        # if isinstance(module, LoraLayer):
        #     if script_args.bf16:
        #         module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        # if "lm_head" in name or "embed_tokens" in name:
        #     if hasattr(module, "weight"):
        #         if script_args.bf16 and module.weight.dtype == torch.float32:
        #             module = module.to(torch.bfloat16)

    trainer.train()

    logger.info("Finished training, saving model...")
    trainer.save_model(config.project_name)

    model_card = utils.create_model_card()

    # save model card to output directory as README.md
    with open(f"{config.project_name}/README.md", "w") as f:
        f.write(model_card)

    if config.use_peft and config.merge_adapter:
        logger.info("Merging adapter weights...")
        try:
            utils.merge_adapter(
                base_model_path=config.model,
                target_model_path=config.project_name,
                adapter_path=config.project_name,
            )
        except Exception as e:
            logger.warning(f"Failed to merge adapter weights: {e}")
            logger.warning("Skipping adapter merge. Only adapter weights will be saved.")

    if config.push_to_hub:
        if PartialState().process_index == 0:
            logger.info("Pushing model to hub...")
            if os.path.exists(f"{config.project_name}/training_params.json"):
                training_params = json.load(open(f"{config.project_name}/training_params.json"))
                training_params.pop("token")
                json.dump(
                    training_params,
                    open(f"{config.project_name}/training_params.json", "w"),
                )
            api = HfApi(token=config.token)
            api.create_repo(repo_id=config.repo_id, repo_type="model", private=True)
            api.upload_folder(
                folder_path=config.project_name,
                repo_id=config.repo_id,
                repo_type="model",
            )

    if PartialState().process_index == 0:
        if "SPACE_ID" in os.environ:
            # shut down the space
            logger.info("Pausing space...")
            api = HfApi(token=config.token)
            api.pause_space(repo_id=os.environ["SPACE_ID"])

        if "ENDPOINT_ID" in os.environ:
            # shut down the endpoint
            logger.info("Pausing endpoint...")
            utils.pause_endpoint(config)


if __name__ == "__main__":
    args = parse_args()
    training_config = json.load(open(args.training_config))
    config = LLMTrainingParams(**training_config)
    train(config)
