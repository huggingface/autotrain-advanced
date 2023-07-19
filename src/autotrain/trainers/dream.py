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
from diffusers.utils.import_utils import is_xformers_available
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

    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=config.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
    )

    if config.train_text_encoder and config.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if config.train_text_encoder and config.xl:
        raise ValueError("XL does not support training the text encoder, yet.")

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    set_seed(config.seed)

    # Generate class images if prior preservation is enabled.
    if config.with_prior_preservation:
        class_images_dir = Path(config.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < config.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if config.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif config.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif config.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                config.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=config.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = config.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = utils.PromptDataset(config.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=config.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True, token=config.hub_token
            ).repo_id

    tokenizers = []
    if config.xl:
        tokenizers.append(
            AutoTokenizer.from_pretrained(
                config.pretrained_model_name_or_path, subfolder="tokenizer", revision=config.revision, use_fast=False
            )
        )
        tokenizers.append(
            AutoTokenizer.from_pretrained(
                config.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=config.revision, use_fast=False
            )
        )
    else:
        tokenizers.append(
            AutoTokenizer.from_pretrained(
                config.pretrained_model_name_or_path, revision=config.revision, use_fast=False
            )
        )

    cls_text_encoders = []
    cls_text_encoders.append(
        utils.import_model_class_from_model_name_or_path(config.pretrained_model_name_or_path, config.revision)
    )
    if config.xl:
        cls_text_encoders.append(
            utils.import_model_class_from_model_name_or_path(
                config.pretrained_model_name_or_path, config.revision, subfolder="text_encoder_2"
            )
        )

    text_encoders = []
    for idx, cls_text_encoder in enumerate(cls_text_encoders):
        if idx > 0:
            subfolder = f"text_encoder_{idx + 1}"
        else:
            subfolder = "text_encoder"
        cls_text_encoder.from_pretrained(
            config.pretrained_model_name_or_path, subfolder=subfolder, revision=config.revision
        )
        text_encoders.append(cls_text_encoder)

    try:
        vae = AutoencoderKL.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="vae", revision=config.revision
        )
    except OSError:
        vae = None

    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="unet", revision=config.revision
    )

    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")

    # TODO: non-peft version
    if vae is not None:
        vae.requires_grad_(False)
    for text_encoder in text_encoders:
        text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if vae is not None:
        vae.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device, dtype=weight_dtype)
    for text_encoder in text_encoders:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

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

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_class = (
                LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
            )

        module = lora_attn_processor_class(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        unet_lora_attn_procs[name] = module
        unet_lora_parameters.extend(module.parameters())

    unet.set_attn_processor(unet_lora_attn_procs)

    if config.train_text_encoder:
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        text_lora_parameters = LoraLoaderMixin._modify_text_encoder(
            text_encoder, dtype=torch.float32, rank=config.rank
        )

    def save_model_hook(models, weights, output_dir):
        # there are only two options here. Either are just the unet attn processor layers
        # or there are the unet and text encoder atten layers
        unet_lora_layers_to_save = None
        text_encoder_lora_layers_to_save = None

        for model in models:
            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_lora_layers_to_save = utils.unet_attn_processors_state_dict(model)
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                text_encoder_lora_layers_to_save = text_encoder_lora_state_dict(model)
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

        LoraLoaderMixin.save_lora_weights(
            output_dir,
            unet_lora_layers=unet_lora_layers_to_save,
            text_encoder_lora_layers=text_encoder_lora_layers_to_save,
        )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                text_encoder_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alpha = LoraLoaderMixin.lora_state_dict(input_dir)
        LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alpha=network_alpha, unet=unet_)
        LoraLoaderMixin.load_lora_into_text_encoder(
            lora_state_dict, network_alpha=network_alpha, text_encoder=text_encoder_
        )

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

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = (
        itertools.chain(unet_lora_parameters, text_lora_parameters)
        if config.train_text_encoder
        else unet_lora_parameters
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    if config.pre_compute_text_embeddings and not config.xl:

        def compute_text_embeddings(prompt):
            with torch.no_grad():
                text_inputs = utils.tokenize_prompt(
                    tokenizer, prompt, tokenizer_max_length=config.tokenizer_max_length
                )
                prompt_embeds = utils.encode_prompt(
                    text_encoder,
                    text_inputs.input_ids,
                    text_inputs.attention_mask,
                    text_encoder_use_attention_mask=config.text_encoder_use_attention_mask,
                )

            return prompt_embeds

        pre_computed_encoder_hidden_states = compute_text_embeddings(config.instance_prompt)
        validation_prompt_negative_prompt_embeds = compute_text_embeddings("")

        if config.validation_prompt is not None:
            validation_prompt_encoder_hidden_states = compute_text_embeddings(config.validation_prompt)
        else:
            validation_prompt_encoder_hidden_states = None

        if config.instance_prompt is not None:
            pre_computed_instance_prompt_encoder_hidden_states = compute_text_embeddings(config.instance_prompt)
        else:
            pre_computed_instance_prompt_encoder_hidden_states = None

        text_encoder = None
        tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()
        instance_prompt_hidden_states = None
        instance_unet_added_conditions = None
        class_prompt_hidden_states = None
        class_unet_added_conditions = None

    elif config.xl:

        def compute_embeddings(prompt, text_encoders, tokenizers):
            original_size = (config.resolution, config.resolution)
            target_size = (config.resolution, config.resolution)
            crops_coords_top_left = (config.crops_coords_top_left_h, config.crops_coords_top_left_w)

            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = utils.encode_prompt_xl(text_encoders, tokenizers, prompt)
                add_text_embeds = pooled_prompt_embeds

                # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                add_time_ids = list(original_size + crops_coords_top_left + target_size)
                add_time_ids = torch.tensor([add_time_ids])

                prompt_embeds = prompt_embeds.to(accelerator.device)
                add_text_embeds = add_text_embeds.to(accelerator.device)
                add_time_ids = add_time_ids.to(accelerator.device, dtype=prompt_embeds.dtype)
                unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

            return prompt_embeds, unet_added_cond_kwargs

        instance_prompt_hidden_states, instance_unet_added_conditions = compute_embeddings(
            config.instance_prompt, text_encoders, tokenizers
        )

        class_prompt_hidden_states, class_unet_added_conditions = None, None
        if config.with_prior_preservation:
            class_prompt_hidden_states, class_unet_added_conditions = compute_embeddings(
                config.class_prompt, text_encoders, tokenizers
            )
        del tokenizers, text_encoders
        gc.collect()
        torch.cuda.empty_cache()
    else:
        pre_computed_encoder_hidden_states = None
        validation_prompt_encoder_hidden_states = None
        validation_prompt_negative_prompt_embeds = None
        pre_computed_instance_prompt_encoder_hidden_states = None
        class_prompt_hidden_states = None
        class_unet_added_conditions = None

    train_dataset = utils.DreamBoothDataset(
        instance_data_root=config.instance_data_dir,
        instance_prompt=config.instance_prompt,
        class_data_root=config.class_data_dir if config.with_prior_preservation else None,
        class_prompt=config.class_prompt,
        class_num=config.num_class_images,
        tokenizer=tokenizer,
        size=config.resolution,
        center_crop=config.center_crop,
        encoder_hidden_states=pre_computed_encoder_hidden_states,
        instance_prompt_encoder_hidden_states=pre_computed_instance_prompt_encoder_hidden_states,
        tokenizer_max_length=config.tokenizer_max_length,
        instance_prompt_hidden_states=instance_prompt_hidden_states,
        class_prompt_hidden_states=class_prompt_hidden_states,
        instance_unet_added_conditions=instance_unet_added_conditions,
        class_unet_added_conditions=class_unet_added_conditions,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: utils.collate_fn(examples, config),
        num_workers=config.dataloader_num_workers,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
        num_cycles=config.lr_num_cycles,
        power=config.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if config.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if overrode_max_train_steps:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(config))

    # Train!
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            path = os.path.basename(config.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * config.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * config.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, config.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, config.num_train_epochs):
        unet.train()
        if config.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if config.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % config.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

                if vae is not None:
                    # Convert images to latent space
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = model_input * vae.config.scaling_factor
                else:
                    model_input = pixel_values

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz, channels, height, width = model_input.shape
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=model_input.device,
                )
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                if config.xl:
                    # Predict the noise residual
                    model_pred = unet(
                        noisy_model_input,
                        timesteps,
                        batch["input_ids"],
                        added_cond_kwargs=batch["unet_added_conditions"],
                    ).sample
                else:
                    # Get the text embedding for conditioning
                    if config.pre_compute_text_embeddings:
                        encoder_hidden_states = batch["input_ids"]
                    else:
                        encoder_hidden_states = utils.encode_prompt(
                            text_encoder,
                            batch["input_ids"],
                            batch["attention_mask"],
                            text_encoder_use_attention_mask=config.text_encoder_use_attention_mask,
                        )

                    if accelerator.unwrap_model(unet).config.in_channels == channels * 2:
                        noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

                    if config.class_labels_conditioning == "timesteps":
                        class_labels = timesteps
                    else:
                        class_labels = None

                    # Predict the noise residual
                    model_pred = unet(
                        noisy_model_input, timesteps, encoder_hidden_states, class_labels=class_labels
                    ).sample

                    # if model predicts variance, throw away the prediction. we will only train on the
                    # simplified training objective. This means that all schedulers using the fine tuned
                    # model must be configured to use one of the fixed variance variance types.
                    if model_pred.shape[1] == 6:
                        model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if config.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + config.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet_lora_parameters, text_lora_parameters)
                        if config.train_text_encoder
                        else unet_lora_parameters
                    )
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % config.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if config.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(config.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= config.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(config.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= config.max_train_steps:
                break

        if accelerator.is_main_process:
            if config.validation_prompt is not None and epoch % config.validation_epochs == 0:
                logger.info(
                    f"Running validation... \n Generating {config.num_validation_images} images with prompt:"
                    f" {config.validation_prompt}."
                )
                # create pipeline
                pipeline = DiffusionPipeline.from_pretrained(
                    config.pretrained_model_name_or_path,
                    unet=accelerator.unwrap_model(unet),
                    text_encoder=None
                    if config.pre_compute_text_embeddings
                    else accelerator.unwrap_model(text_encoder),
                    revision=config.revision,
                    torch_dtype=weight_dtype,
                )

                # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
                scheduler_args = {}

                if "variance_type" in pipeline.scheduler.config:
                    variance_type = pipeline.scheduler.config.variance_type

                    if variance_type in ["learned", "learned_range"]:
                        variance_type = "fixed_small"

                    scheduler_args["variance_type"] = variance_type

                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipeline.scheduler.config, **scheduler_args
                )

                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = (
                    torch.Generator(device=accelerator.device).manual_seed(config.seed) if config.seed else None
                )
                if config.pre_compute_text_embeddings:
                    pipeline_args = {
                        "prompt_embeds": validation_prompt_encoder_hidden_states,
                        "negative_prompt_embeds": validation_prompt_negative_prompt_embeds,
                    }
                else:
                    pipeline_args = {"prompt": config.validation_prompt}

                if config.validation_images is None:
                    images = []
                    for _ in range(config.num_validation_images):
                        with torch.cuda.amp.autocast():
                            image = pipeline(**pipeline_args, generator=generator).images[0]
                            images.append(image)
                else:
                    images = []
                    for image in config.validation_images:
                        image = Image.open(image)
                        with torch.cuda.amp.autocast():
                            image = pipeline(**pipeline_args, image=image, generator=generator).images[0]
                        images.append(image)

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")

                del pipeline
                torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_layers = utils.unet_attn_processors_state_dict(unet)

        if text_encoder is not None and config.train_text_encoder:
            text_encoder = accelerator.unwrap_model(text_encoder)
            text_encoder = text_encoder.to(torch.float32)
            text_encoder_lora_layers = text_encoder_lora_state_dict(text_encoder)
        else:
            text_encoder_lora_layers = None

        LoraLoaderMixin.save_lora_weights(
            save_directory=config.output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
        )

        # Final inference
        # Load previous pipeline
        pipeline = DiffusionPipeline.from_pretrained(
            config.pretrained_model_name_or_path, revision=config.revision, torch_dtype=weight_dtype
        )

        # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
        scheduler_args = {}

        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

        pipeline = pipeline.to(accelerator.device)

        # load attention processors
        pipeline.load_lora_weights(config.output_dir, weight_name="pytorch_lora_weights.bin")

        # run inference
        images = []
        if config.validation_prompt and config.num_validation_images > 0:
            generator = torch.Generator(device=accelerator.device).manual_seed(config.seed) if config.seed else None
            images = [
                pipeline(config.validation_prompt, num_inference_steps=25, generator=generator).images[0]
                for _ in range(config.num_validation_images)
            ]

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")

        if config.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=config.pretrained_model_name_or_path,
                train_text_encoder=config.train_text_encoder,
                prompt=config.instance_prompt,
                repo_folder=config.output_dir,
                pipeline=pipeline,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=config.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()
