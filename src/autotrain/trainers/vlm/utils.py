import os

import torch
from accelerate import PartialState
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, PaliGemmaForConditionalGeneration

from autotrain import logger
from autotrain.trainers.common import (
    ALLOW_REMOTE_CODE,
    LossLoggingCallback,
    TrainStartCallback,
    UploadLogs,
    pause_space,
    remove_autotrain_data,
    save_training_params,
)


TARGET_MODULES = {}

SUPPORTED_MODELS = [
    "PaliGemmaForConditionalGeneration",
    "Florence2ForConditionalGeneration",
]

MODEL_CARD = """
---
tags:
- autotrain
- text-generation-inference
- image-text-to-text
- text-generation{peft}
library_name: transformers{base_model}
license: other{dataset_tag}
---

# Model Trained Using AutoTrain

This model was trained using AutoTrain. For more information, please visit [AutoTrain](https://hf.co/docs/autotrain).

# Usage

"""


def get_target_modules(config):
    if config.target_modules is None:
        return TARGET_MODULES.get(config.model)
    if config.target_modules.strip() == "":
        return TARGET_MODULES.get(config.model)
    if config.target_modules.strip().lower() == "all-linear":
        return "all-linear"
    return config.target_modules.split(",")


def create_model_card(config):
    if config.peft:
        peft = "\n- peft"
    else:
        peft = ""

    if config.data_path == f"{config.project_name}/autotrain-data" or os.path.isdir(config.data_path):
        dataset_tag = ""
    else:
        dataset_tag = f"\ndatasets:\n- {config.data_path}"

    if os.path.isdir(config.model):
        base_model = ""
    else:
        base_model = f"\nbase_model: {config.model}"

    model_card = MODEL_CARD.format(
        dataset_tag=dataset_tag,
        peft=peft,
        base_model=base_model,
    )
    return model_card.strip()


def check_model_support(config):
    api = HfApi(token=config.token)
    model_info = api.model_info(config.model)
    architectures = model_info.config.get("architectures", [])
    for arch in architectures:
        if arch in SUPPORTED_MODELS:
            return True
    return False


def configure_logging_steps(config, train_data, valid_data):
    logger.info("configuring logging steps")
    if config.logging_steps == -1:
        if config.valid_split is not None:
            logging_steps = int(0.2 * len(valid_data) / config.batch_size)
        else:
            logging_steps = int(0.2 * len(train_data) / config.batch_size)
        if logging_steps == 0:
            logging_steps = 1
        if logging_steps > 25:
            logging_steps = 25
        config.logging_steps = logging_steps
    else:
        logging_steps = config.logging_steps
    logger.info(f"Logging steps: {logging_steps}")
    return logging_steps


def configure_training_args(config, logging_steps):
    logger.info("configuring training args")
    training_args = dict(
        output_dir=config.project_name,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        eval_strategy=config.eval_strategy if config.valid_split is not None else "no",
        logging_steps=logging_steps,
        save_total_limit=config.save_total_limit,
        save_strategy=config.eval_strategy if config.valid_split is not None else "no",
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

    if not config.disable_gradient_checkpointing:
        if config.peft and config.quantization in ("int4", "int8"):
            training_args["gradient_checkpointing_kwargs"] = {"use_reentrant": True}
        else:
            training_args["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    if config.mixed_precision == "fp16":
        training_args["fp16"] = True
    if config.mixed_precision == "bf16":
        training_args["bf16"] = True

    return training_args


def get_callbacks(config):
    callbacks = [UploadLogs(config=config), LossLoggingCallback(), TrainStartCallback()]
    return callbacks


def get_model(config):
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

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            config.model,
            config=model_config,
            token=config.token,
            quantization_config=bnb_config,
            trust_remote_code=ALLOW_REMOTE_CODE,
        )
    else:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            config.model,
            config=model_config,
            token=config.token,
            trust_remote_code=ALLOW_REMOTE_CODE,
        )

    logger.info(f"model dtype: {model.dtype}")

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
            target_modules=get_target_modules(config),
        )
        model = get_peft_model(model, peft_config)

    for param in model.vision_tower.parameters():
        param.requires_grad = False

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

    return model


def post_training_steps(config, trainer):
    logger.info("Finished training, saving model...")
    trainer.model.config.use_cache = True
    trainer.save_model(config.project_name)

    model_card = create_model_card(config)

    # save model card to output directory as README.md
    with open(f"{config.project_name}/README.md", "w", encoding="utf-8") as f:
        f.write(model_card)

    # if config.peft and config.merge_adapter:
    #     logger.info("Merging adapter weights...")
    #     try:
    #         merge_adapter(
    #             base_model_path=config.model,
    #             target_model_path=config.project_name,
    #             adapter_path=config.project_name,
    #         )
    #         # remove adapter weights: adapter_*
    #         for file in os.listdir(config.project_name):
    #             if file.startswith("adapter_"):
    #                 os.remove(f"{config.project_name}/{file}")
    #     except Exception as e:
    #         logger.warning(f"Failed to merge adapter weights: {e}")
    #         logger.warning("Skipping adapter merge. Only adapter weights will be saved.")

    if config.push_to_hub:
        if PartialState().process_index == 0:
            # remove data folder
            remove_autotrain_data(config)
            logger.info("Pushing model to hub...")
            save_training_params(config)
            api = HfApi(token=config.token)
            api.create_repo(
                repo_id=f"{config.username}/{config.project_name}", repo_type="model", private=True, exist_ok=True
            )
            api.upload_folder(
                folder_path=config.project_name,
                repo_id=f"{config.username}/{config.project_name}",
                repo_type="model",
            )

    if PartialState().process_index == 0:
        pause_space(config)
