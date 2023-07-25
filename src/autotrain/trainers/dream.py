import gc
import hashlib
import itertools
import math
import os
import shutil
from pathlib import Path

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin, text_encoder_lora_state_dict
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from diffusers.optimization import get_scheduler
from huggingface_hub import create_repo, upload_folder
from loguru import logger
from packaging import version
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

from autotrain.trainers import dreambooth_utils as utils


def train(config):
    if isinstance(config, dict):
        config = utils.DreamBoothTrainingParams(**config)

    print(config)
    config.instance_prompt = str(config.instance_prompt).strip()

    config.mixed_precision = "fp16"

    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=config.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
    )

    if config.model_name in utils.XL_MODELS:
        config.xl = True

    if config.train_text_encoder and config.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    set_seed(config.seed)

    # Generate class images if prior preservation is enabled.
    if config.with_prior_preservation:
        utils.setup_prior_preservation(accelerator, config)

    # Handle the repository creation
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True, token=config.hub_token
            ).repo_id

    tokenizers, text_encoders, vae, unet, noise_scheduler, weight_dtype = utils.load_model_components(
        config, accelerator
    )

    utils.enable_xformers(unet, config)

    unet_lora_attn_procs = {}
    unet_lora_parameters = []
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        # if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
        #     lora_attn_processor_class = LoRAAttnAddedKVProcessor
        # else:
        lora_attn_processor_class = (
            LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
        )

        module = lora_attn_processor_class(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        unet_lora_attn_procs[name] = module
        unet_lora_parameters.extend(module.parameters())

    unet.set_attn_processor(unet_lora_attn_procs)

    text_lora_parameters = []
    if config.train_text_encoder:
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        text_lora_parameters = [
            LoraLoaderMixin._modify_text_encoder(_text_encoder, dtype=torch.float32) for _text_encoder in text_encoders
        ]

    def save_model_hook(models, weights, output_dir):
        # there are only two options here. Either are just the unet attn processor layers
        # or there are the unet and text encoder atten layers
        unet_lora_layers_to_save = None
        text_encoder_lora_layers_to_save = []

        for model in models:
            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_lora_layers_to_save = utils.unet_attn_processors_state_dict(model)

            for _text_encoder in text_encoders:
                if isinstance(model, type(accelerator.unwrap_model(_text_encoder))):
                    text_encoder_lora_layers_to_save.append(text_encoder_lora_state_dict(model))

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

        if len(text_encoder_lora_layers_to_save) == 0:
            LoraLoaderMixin.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=None,
            )
        elif len(text_encoder_lora_layers_to_save) == 1:
            LoraLoaderMixin.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_lora_layers_to_save[0],
            )
        elif len(text_encoder_lora_layers_to_save) == 2:
            StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_lora_layers_to_save[0],
                text_encoder_2_lora_layers=text_encoder_lora_layers_to_save[1],
            )
        else:
            raise ValueError("unexpected number of text encoders")

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoders_ = []

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            # elif isinstance(model, type(accelerator.unwrap_model(text_encoder))):
            #     text_encoder_ = model
            # else:
            #     raise ValueError(f"unexpected save model: {model.__class__}")
            for _text_encoder in text_encoders:
                if isinstance(model, type(accelerator.unwrap_model(_text_encoder))):
                    text_encoders_.append(model)

        lora_state_dict, network_alpha = LoraLoaderMixin.lora_state_dict(input_dir)
        LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alpha=network_alpha, unet=unet_)

        if len(text_encoders_) == 0:
            LoraLoaderMixin.load_lora_into_text_encoder(
                lora_state_dict,
                network_alpha=network_alpha,
                text_encoder=None,
            )
        elif len(text_encoders_) == 1:
            LoraLoaderMixin.load_lora_into_text_encoder(
                lora_state_dict,
                network_alpha=network_alpha,
                text_encoder=text_encoders_[0],
            )
        elif len(text_encoders_) == 2:
            LoraLoaderMixin.load_lora_into_text_encoder(
                lora_state_dict,
                network_alpha=network_alpha,
                text_encoder=text_encoders_[0],
            )
            LoraLoaderMixin.load_lora_into_text_encoder(
                lora_state_dict,
                network_alpha=network_alpha,
                text_encoder=text_encoders_[1],
            )
        else:
            raise ValueError("unexpected number of text encoders")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.scale_lr:
        config.learning_rate = (
            config.learning_rate
            * config.gradient_accumulation_steps
            * config.train_batch_size
            * accelerator.num_processes
        )

    optimizer = utils.get_optimizer(config, unet_lora_parameters, text_lora_parameters)

    # (
    #     pre_computed_encoder_hidden_states,
    #     validation_prompt_encoder_hidden_states,
    #     validation_prompt_negative_prompt_embeds,
    #     pre_computed_instance_prompt_encoder_hidden_states,
    #     instance_prompt_hidden_states,
    #     instance_unet_added_conditions,
    #     class_prompt_hidden_states,
    #     class_unet_added_conditions,
    # ) = utils.compute_embeddings(
    #     config=config,
    #     accelerator=accelerator,
    #     text_encoders=text_encoders,
    #     tokenizers=tokenizers,
    # )
    if config.xl:
        train_dataset = utils.DreamBoothDatasetXL(
            instance_data_root=config.image_path,
            class_data_root=config.class_image_path if config.with_prior_preservation else None,
            class_num=config.num_class_images,
            size=config.resolution,
            center_crop=config.center_crop,
            # two more args
        )
    else:
        train_dataset = utils.DreamBoothDataset(
            instance_data_root=config.image_path,
            instance_prompt=config.instance_prompt,
            class_data_root=config.class_image_path if config.with_prior_preservation else None,
            class_prompt=config.class_prompt,
            class_num=config.num_class_images,
            tokenizer=tokenizers[0],
            size=config.resolution,
            center_crop=config.center_crop,
            tokenizer_max_length=config.tokenizer_max_length,
            config=config,
            # two more args
        )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: utils.collate_fn(examples, config),
        num_workers=config.dataloader_num_workers,
    )
    trainer = utils.Trainer(
        unet=unet,
        vae=vae,
        train_dataloader=train_dataloader,
        text_encoders=text_encoders,
        config=config,
        optimizer=optimizer,
        accelerator=accelerator,
        noise_scheduler=noise_scheduler,
        train_dataset=train_dataset,
        weight_dtype=weight_dtype,
        text_lora_parameters=text_lora_parameters,
        unet_lora_parameters=unet_lora_parameters,
        tokenizers=tokenizers,
    )
    trainer.train()
