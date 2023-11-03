import itertools
import math
import os
import shutil

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from diffusers.loaders import LoraLoaderMixin, text_encoder_lora_state_dict
from diffusers.optimization import get_scheduler
from huggingface_hub import create_repo, upload_folder
from tqdm import tqdm

from autotrain import logger
from autotrain.trainers.dreambooth import utils


class Trainer:
    def __init__(
        self,
        unet,
        vae,
        train_dataloader,
        train_dataset,
        text_encoders,
        config,
        optimizer,
        accelerator,
        noise_scheduler,
        weight_dtype,
        text_lora_parameters,
        unet_lora_parameters,
        tokenizers,
    ):
        self.train_dataloader = train_dataloader
        self.config = config
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.unet = unet
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.train_dataset = train_dataset
        self.weight_dtype = weight_dtype
        self.text_lora_parameters = text_lora_parameters
        self.unet_lora_parameters = unet_lora_parameters
        self.tokenizers = tokenizers
        self.text_encoders = text_encoders

        if self.config.xl:
            self._setup_xl()

        self.text_encoder1 = text_encoders[0]
        self.text_encoder2 = None
        if len(text_encoders) == 2:
            self.text_encoder2 = text_encoders[1]

        overrode_max_train_steps = False
        self.num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation)
        if self.config.num_steps is None:
            self.config.num_steps = self.config.epochs * self.num_update_steps_per_epoch
            overrode_max_train_steps = True

        self.scheduler = get_scheduler(
            self.config.scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.config.num_steps * self.accelerator.num_processes,
            num_cycles=self.config.num_cycles,
            power=self.config.lr_power,
        )

        if self.config.train_text_encoder:
            if len(text_encoders) == 1:
                (
                    self.unet,
                    self.text_encoder1,
                    self.optimizer,
                    self.train_dataloader,
                    self.scheduler,
                ) = self.accelerator.prepare(
                    self.unet,
                    self.text_encoder1,
                    self.optimizer,
                    self.train_dataloader,
                    self.scheduler,
                )
            elif len(text_encoders) == 2:
                (
                    self.unet,
                    self.text_encoder1,
                    self.text_encoder2,
                    self.optimizer,
                    self.train_dataloader,
                    self.scheduler,
                ) = self.accelerator.prepare(
                    self.unet,
                    self.text_encoder1,
                    self.text_encoder2,
                    self.optimizer,
                    self.train_dataloader,
                    self.scheduler,
                )

        else:
            (
                self.unet,
                self.optimizer,
                self.train_dataloader,
                self.scheduler,
            ) = accelerator.prepare(self.unet, self.optimizer, self.train_dataloader, self.scheduler)

        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.config.gradient_accumulation)
        if overrode_max_train_steps:
            self.config.num_steps = self.config.epochs * self.num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.config.epochs = math.ceil(self.config.num_steps / self.num_update_steps_per_epoch)

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("dreambooth")

        self.total_batch_size = (
            self.config.batch_size * self.accelerator.num_processes * self.config.gradient_accumulation
        )
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(self.train_dataloader)}")
        logger.info(f"  Num Epochs = {self.config.epochs}")
        logger.info(f"  Instantaneous batch size per device = {config.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation}")
        logger.info(f"  Total optimization steps = {self.config.num_steps}")
        logger.info(f"  Training config = {self.config}")
        self.global_step = 0
        self.first_epoch = 0

        if config.resume_from_checkpoint:
            self._resume_from_checkpoint()

    def compute_text_embeddings(self, prompt):
        logger.info(f"Computing text embeddings for prompt: {prompt}")
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = utils.encode_prompt_xl(self.text_encoders, self.tokenizers, prompt)
            prompt_embeds = prompt_embeds.to(self.accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(self.accelerator.device)
        return prompt_embeds, pooled_prompt_embeds

    def compute_time_ids(self):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        original_size = (self.config.resolution, self.config.resolution)
        target_size = (self.config.resolution, self.config.resolution)
        # crops_coords_top_left = (self.config.crops_coords_top_left_h, self.config.crops_coords_top_left_w)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(self.accelerator.device, dtype=self.weight_dtype)
        return add_time_ids

    def _setup_xl(self):
        # Handle instance prompt.
        instance_time_ids = self.compute_time_ids()
        if not self.config.train_text_encoder:
            (
                instance_prompt_hidden_states,
                instance_pooled_prompt_embeds,
            ) = self.compute_text_embeddings(self.config.prompt)

        # Handle class prompt for prior-preservation.
        if self.config.prior_preservation:
            class_time_ids = self.compute_time_ids()
            if not self.config.train_text_encoder:
                (
                    class_prompt_hidden_states,
                    class_pooled_prompt_embeds,
                ) = self.compute_text_embeddings(self.config.class_prompt)

        self.add_time_ids = instance_time_ids
        if self.config.prior_preservation:
            self.add_time_ids = torch.cat([self.add_time_ids, class_time_ids], dim=0)

        if not self.config.train_text_encoder:
            self.prompt_embeds = instance_prompt_hidden_states
            self.unet_add_text_embeds = instance_pooled_prompt_embeds
            if self.config.prior_preservation:
                self.prompt_embeds = torch.cat([self.prompt_embeds, class_prompt_hidden_states], dim=0)
                self.unet_add_text_embeds = torch.cat([self.unet_add_text_embeds, class_pooled_prompt_embeds], dim=0)
        else:
            self.tokens_one = utils.tokenize_prompt(self.tokenizers[0], self.config.prompt).input_ids
            self.tokens_two = utils.tokenize_prompt(self.tokenizers[1], self.config.prompt).input_ids
            if self.config.prior_preservation:
                class_tokens_one = utils.tokenize_prompt(self.tokenizers[0], self.config.class_prompt).input_ids
                class_tokens_two = utils.tokenize_prompt(self.tokenizers[1], self.config.class_prompt).input_ids
                self.tokens_one = torch.cat([self.tokens_one, class_tokens_one], dim=0)
                self.tokens_two = torch.cat([self.tokens_two, class_tokens_two], dim=0)

    def _resume_from_checkpoint(self):
        if self.config.resume_from_checkpoint != "latest":
            path = os.path.basename(self.config.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(self.config.project_name)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            self.accelerator.print(
                f"Checkpoint '{self.config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            self.config.resume_from_checkpoint = None
        else:
            self.accelerator.print(f"Resuming from checkpoint {path}")
            self.accelerator.load_state(os.path.join(self.config.project_name, path))
            self.global_step = int(path.split("-")[1])

            resume_global_step = self.global_step * self.config.gradient_accumulation
            self.first_epoch = self.global_step // self.num_update_steps_per_epoch
            self.resume_step = resume_global_step % (
                self.num_update_steps_per_epoch * self.config.gradient_accumulation
            )

    def _calculate_loss(self, model_pred, noise, model_input, timesteps):
        if model_pred.shape[1] == 6 and not self.config.xl:
            model_pred, _ = torch.chunk(model_pred, 2, dim=1)

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        if self.config.prior_preservation:
            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Compute prior loss
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

            # Add the prior loss to the instance loss.
            loss = loss + self.config.prior_loss_weight * prior_loss
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return loss

    def _clip_gradients(self):
        if self.accelerator.sync_gradients:
            if len(self.text_lora_parameters) == 0:
                params_to_clip = self.unet_lora_parameters
            elif len(self.text_lora_parameters) == 1:
                params_to_clip = itertools.chain(self.unet_lora_parameters, self.text_lora_parameters[0])
            elif len(self.text_lora_parameters) == 2:
                params_to_clip = itertools.chain(
                    self.unet_lora_parameters,
                    self.text_lora_parameters[0],
                    self.text_lora_parameters[1],
                )
            else:
                raise ValueError("More than 2 text encoders are not supported.")
            self.accelerator.clip_grad_norm_(params_to_clip, self.config.max_grad_norm)

    def _save_checkpoint(self):
        if self.accelerator.is_main_process:
            if self.global_step % self.config.checkpointing_steps == 0:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if self.config.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(self.config.project_name)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= self.config.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - self.config.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(self.config.project_name, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(self.config.project_name, f"checkpoint-{self.global_step}")
                self.accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

    def _get_model_pred(self, batch, channels, noisy_model_input, timesteps, bsz):
        if self.config.xl:
            elems_to_repeat = bsz // 2 if self.config.prior_preservation else bsz
            if not self.config.train_text_encoder:
                unet_added_conditions = {
                    "time_ids": self.add_time_ids.repeat(elems_to_repeat, 1),
                    "text_embeds": self.unet_add_text_embeds.repeat(elems_to_repeat, 1),
                }
                model_pred = self.unet(
                    noisy_model_input,
                    timesteps,
                    self.prompt_embeds.repeat(elems_to_repeat, 1, 1),
                    added_cond_kwargs=unet_added_conditions,
                ).sample
            else:
                unet_added_conditions = {"time_ids": self.add_time_ids.repeat(elems_to_repeat, 1)}
                prompt_embeds, pooled_prompt_embeds = utils.encode_prompt_xl(
                    text_encoders=self.text_encoders,
                    tokenizers=None,
                    prompt=None,
                    text_input_ids_list=[self.tokens_one, self.tokens_two],
                )
                unet_added_conditions.update({"text_embeds": pooled_prompt_embeds.repeat(elems_to_repeat, 1)})
                prompt_embeds = prompt_embeds.repeat(elems_to_repeat, 1, 1)
                model_pred = self.unet(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                ).sample

        else:
            if self.config.pre_compute_text_embeddings:
                encoder_hidden_states = batch["input_ids"]
            else:
                encoder_hidden_states = utils.encode_prompt(
                    self.text_encoder1,
                    batch["input_ids"],
                    batch["attention_mask"],
                    text_encoder_use_attention_mask=self.config.text_encoder_use_attention_mask,
                )

            if self.accelerator.unwrap_model(self.unet).config.in_channels == channels * 2:
                noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

            if self.config.class_labels_conditioning == "timesteps":
                class_labels = timesteps
            else:
                class_labels = None

            model_pred = self.unet(
                noisy_model_input,
                timesteps,
                encoder_hidden_states,
                class_labels=class_labels,
            ).sample

        return model_pred

    def train(self):
        progress_bar = tqdm(
            range(self.global_step, self.config.num_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")

        for epoch in range(self.first_epoch, self.config.epochs):
            self.unet.train()

            if self.config.train_text_encoder:
                self.text_encoder1.train()
                if self.config.xl:
                    self.text_encoder2.train()

            for step, batch in enumerate(self.train_dataloader):
                # Skip steps until we reach the resumed step
                if self.config.resume_from_checkpoint and epoch == self.first_epoch and step < self.resume_step:
                    if step % self.config.gradient_accumulation == 0:
                        progress_bar.update(1)
                    continue

                with self.accelerator.accumulate(self.unet):
                    if self.config.xl:
                        pixel_values = batch["pixel_values"]
                    else:
                        pixel_values = batch["pixel_values"].to(dtype=self.weight_dtype)

                    if self.vae is not None:
                        # Convert images to latent space
                        model_input = self.vae.encode(pixel_values).latent_dist.sample()
                        model_input = model_input * self.vae.config.scaling_factor
                        model_input = model_input.to(dtype=self.weight_dtype)
                    else:
                        model_input = pixel_values

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    bsz, channels, height, width = model_input.shape
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=model_input.device,
                    )
                    timesteps = timesteps.long()

                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)
                    model_pred = self._get_model_pred(batch, channels, noisy_model_input, timesteps, bsz)
                    loss = self._calculate_loss(model_pred, noise, model_input, timesteps)
                    self.accelerator.backward(loss)

                    self._clip_gradients()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.global_step += 1
                    self._save_checkpoint()

                logs = {
                    "loss": loss.detach().item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=self.global_step)

                if self.global_step >= self.config.num_steps:
                    break

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.unet = self.accelerator.unwrap_model(self.unet)
            self.unet = self.unet.to(torch.float32)
            unet_lora_layers = utils.unet_attn_processors_state_dict(self.unet)
            text_encoder_lora_layers_1 = None
            text_encoder_lora_layers_2 = None

            if self.text_encoder1 is not None and self.config.train_text_encoder:
                text_encoder1 = self.accelerator.unwrap_model(self.text_encoder1)
                text_encoder1 = text_encoder1.to(torch.float32)
                text_encoder_lora_layers_1 = text_encoder_lora_state_dict(text_encoder1)

            if self.text_encoder2 is not None and self.config.train_text_encoder:
                text_encoder2 = self.accelerator.unwrap_model(self.text_encoder2)
                text_encoder2 = text_encoder2.to(torch.float32)
                text_encoder_lora_layers_2 = text_encoder_lora_state_dict(text_encoder2)

            if self.config.xl:
                StableDiffusionXLPipeline.save_lora_weights(
                    save_directory=self.config.project_name,
                    unet_lora_layers=unet_lora_layers,
                    text_encoder_lora_layers=text_encoder_lora_layers_1,
                    text_encoder_2_lora_layers=text_encoder_lora_layers_2,
                    safe_serialization=True,
                )
            else:
                LoraLoaderMixin.save_lora_weights(
                    save_directory=self.config.project_name,
                    unet_lora_layers=unet_lora_layers,
                    text_encoder_lora_layers=text_encoder_lora_layers_1,
                    safe_serialization=True,
                )
        self.accelerator.end_training()

    def push_to_hub(self):
        repo_id = create_repo(
            repo_id=self.config.repo_id,
            exist_ok=True,
            private=True,
            token=self.config.token,
        ).repo_id

        utils.create_model_card(
            repo_id,
            base_model=self.config.model,
            train_text_encoder=self.config.train_text_encoder,
            prompt=self.config.prompt,
            repo_folder=self.config.project_name,
        )
        upload_folder(
            repo_id=repo_id,
            folder_path=self.config.project_name,
            commit_message="End of training",
            ignore_patterns=["step_*", "epoch_*"],
            token=self.config.token,
        )
