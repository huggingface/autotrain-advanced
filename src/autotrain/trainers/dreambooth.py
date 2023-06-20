import gc
import itertools
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfApi
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from autotrain import utils
from autotrain.params import DreamboothParams
from autotrain.utils import LFS_PATTERNS


MARKDOWN = """
---
tags:
- autotrain
- stable-diffusion
- text-to-image
datasets:
- {dataset}
co2_eq_emissions:
  emissions: {co2}
---

# Model Trained Using AutoTrain

- Problem type: Dreambooth
- Model ID: {model_id}
- CO2 Emissions (in grams): {co2:.4f}
"""

SPACE_README = """
---
title: AutoTrain Dreambooth({model_id})
emoji: 😻
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 3.12.0
app_file: app.py
pinned: false
tags:
  - autotrain
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
"""

SPACE_APP = """
import os

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPE = StableDiffusionPipeline.from_pretrained(
    "model/",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)
PIPE = PIPE.to(DEVICE)


def generate_image(prompt, negative_prompt, image_size, scale, steps, seed):
    image_size = int(image_size) if image_size else int({img_size})
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    images = PIPE(
        prompt,
        negative_prompt=negative_prompt,
        width=image_size,
        height=image_size,
        num_inference_steps=steps,
        guidance_scale=scale,
        num_images_per_prompt=1,
        generator=generator,
    ).images[0]
    return images


gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", lines=5, max_lines=5),
        gr.Textbox(label="Negative prompt (optional)", lines=5, max_lines=5),
        gr.Textbox(label="Image size (optional)", lines=1, max_lines=1),
        gr.Slider(1, maximum=20, value=7.5, step=0.5, label="Scale"),
        gr.Slider(1, 150, 50, label="Steps"),
        gr.Slider(minimum=1, step=1, maximum=999999999999999999, randomize=True, label="Seed"),
    ],
    outputs="image",
    title="Dreambooth - Powered by AutoTrain",
    description="Model:{model_id}, concept prompts: {concept_prompts}. Tip: Switch to GPU hardware in settings to make inference superfast!",
).launch()
"""

SPACE_REQUIREMENTS = """
--extra-index-url https://download.pytorch.org/whl/cu113
torch==1.12.1+cu113
torchvision==0.13.1+cu113
accelerate
transformers
git+https://github.com/huggingface/diffusers.git
"""


def create_model_card(dataset_id: str, model_id: str, co2: float):
    co2 = co2 * 1000 if co2 is not None else 0
    logger.info("Generating markdown for dreambooth")
    markdown = MARKDOWN.strip().format(
        model_id=model_id,
        dataset=dataset_id,
        co2=co2,
    )
    return markdown


@dataclass
class TrainingArgs:
    pretrained_model_name_or_path: str
    instance_data_dir: str
    revision: Optional[str] = None
    class_data_dir: Optional[str] = None
    tokenizer_name: Optional[str] = None
    class_prompt: str = ""
    with_prior_preservation: bool = False
    prior_loss_weight: float = 1.0
    num_class_images: int = 100
    output_dir: str = ""
    seed: int = 42
    resolution: int = 512
    center_crop: bool = False
    train_text_encoder: bool = True
    train_batch_size: int = 4
    sample_batch_size: int = 4
    num_train_epochs: int = 1
    max_train_steps: int = 5000
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 5e-6
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    logging_dir: str = "logs"
    mixed_precision: str = "no"
    stop_text_encoder_training: int = 1000000
    cache_latents: bool = False


def pad_image(image):
    w, h = image.size
    if w == h:
        return image
    elif w > h:
        new_image = Image.new(image.mode, (w, w), (0, 0, 0))
        new_image.paste(image, (0, (w - h) // 2))
        return new_image
    else:
        new_image = Image.new(image.mode, (h, h), (0, 0, 0))
        new_image.paste(image, ((h - w) // 2, 0))
        return new_image


def process_images(data_path, job_config):
    # create processed_data folder in data_path
    processed_data_path = os.path.join(data_path, "processed_data")
    os.makedirs(processed_data_path, exist_ok=True)
    # find all folders in data_path that start with "concept"
    concept_folders = [f for f in os.listdir(data_path) if f.startswith("concept")]
    concept_prompts = json.load(open(os.path.join(data_path, "prompts.json")))

    for concept_folder in concept_folders:
        concept_folder_path = os.path.join(data_path, concept_folder)
        # find all images in concept_folder_path
        ALLOWED_EXTENSIONS = ["jpg", "png", "jpeg"]
        images = [
            f for f in os.listdir(concept_folder_path) if any(f.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)
        ]
        for image_index, image in enumerate(images):
            image_path = os.path.join(concept_folder_path, image)
            img = Image.open(image_path)
            img = pad_image(img)
            img = img.resize((job_config.image_size, job_config.image_size))
            img = img.convert("RGB")
            processed_filename = f"{concept_prompts[concept_folder]}_{image_index}.jpg"
            img.save(os.path.join(processed_data_path, processed_filename), format="JPEG", quality=100)
    return concept_prompts


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        tokenizer,
        size,
        class_data_root=None,
        class_prompt=None,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.class_data_root = class_data_root

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            random.shuffle(self.class_images_path)
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        path = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        filename = Path(path).stem
        pt = "".join([i for i in filename if not i.isdigit()])
        pt = pt.replace("_", " ")
        pt = pt.replace("(", "")
        pt = pt.replace(")", "")
        pt = pt.replace("-", "")
        instance_prompt = pt
        sys.stdout.write(" [0;32m" + instance_prompt + " [0m")
        sys.stdout.flush()

        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class LatentsDataset(Dataset):
    def __init__(self, latents_cache, text_encoder_cache):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        return self.latents_cache[index], self.text_encoder_cache[index]


def collate_fn(examples, tokenizer, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch


def run_training(args):
    logger.info(args)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    image.save(class_images_dir / f"{example['index'][i] + cur_class_images}.jpg")

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, tokenizer, args.with_prior_preservation),
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.cache_latents:
        latents_cache = []
        text_encoder_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(
                    accelerator.device, non_blocking=True, dtype=weight_dtype
                )
                batch["input_ids"] = batch["input_ids"].to(accelerator.device, non_blocking=True)
                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)
                if args.train_text_encoder:
                    text_encoder_cache.append(batch["input_ids"])
                else:
                    text_encoder_cache.append(text_encoder(batch["input_ids"])[0])
        train_dataset = LatentsDataset(latents_cache, text_encoder_cache)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=True
        )

        del vae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    first_epoch = 0

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    if args.cache_latents:
                        latents_dist = batch[0][0]
                    else:
                        latents_dist = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist
                    latents = latents_dist.sample() * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                if args.cache_latents:
                    if args.train_text_encoder:
                        encoder_hidden_states = text_encoder(batch[0][1])[0]
                    else:
                        encoder_hidden_states = batch[0][1]
                else:
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if args.train_text_encoder and global_step == args.stop_text_encoder_training and global_step >= 30:
                if accelerator.is_main_process:
                    logger.info("Freezing the text_encoder ...")
                    frz_dir = args.output_dir + "/text_encoder_frozen"
                    if os.path.exists(frz_dir):
                        subprocess.call("rm -r " + frz_dir, shell=True)
                    os.mkdir(frz_dir)
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet),
                        text_encoder=accelerator.unwrap_model(text_encoder),
                    )
                    pipeline.text_encoder.save_pretrained(frz_dir)
                    try:
                        pipeline.text_encoder.save_pretrained(frz_dir, safe_serialization=True)
                    except Exception as e:
                        logger.error("Failed to save the text_encoder with safe serialization: " + str(e))

        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
        )
        frz_dir = args.output_dir + "/text_encoder_frozen"
        pipeline.save_pretrained(args.output_dir)
        try:
            pipeline.save_pretrained(args.output_dir, safe_serialization=True)
        except Exception as e:
            logger.error("Failed to save the pipeline with safe serialization: " + str(e))
        if args.train_text_encoder and os.path.exists(frz_dir):
            subprocess.call("mv -f " + frz_dir + "/*.* " + args.output_dir + "/text_encoder", shell=True)
            subprocess.call("rm -r " + frz_dir, shell=True)

    accelerator.end_training()
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()


@utils.job_watcher
def train(co2_tracker, payload, huggingface_token, model_path):
    data_repo_path = f"{payload['username']}/autotrain-data-{payload['proj_name']}"
    data_path = "/tmp/data"
    data_repo = utils.clone_hf_repo(
        local_dir=data_path,
        repo_url="https://huggingface.co/datasets/" + data_repo_path,
        token=huggingface_token,
    )
    data_repo.git_pull()

    job_config = payload["config"]["params"][0]
    job_config["model_name"] = payload["config"]["hub_model"]

    model_name = job_config["model_name"]
    # device = job_config.get("device", "cuda")
    del job_config["model_name"]
    if "device" in job_config:
        del job_config["device"]
    job_config = DreamboothParams(**job_config)

    logger.info("Create model repo")
    project_name = payload["proj_name"]
    repo_name = f"autotrain-{project_name}"
    repo_user = payload["username"]
    repo_url = HfApi().create_repo(
        repo_id=f"{repo_user}/{repo_name}", token=huggingface_token, exist_ok=True, private=True
    )
    if len(repo_url.strip()) == 0:
        repo_url = f"https://huggingface.co/{repo_user}/{repo_name}"

    space_repo_url = HfApi().create_repo(
        repo_id=f"{repo_user}/{repo_name}",
        token=huggingface_token,
        exist_ok=True,
        private=True,
        repo_type="space",
        space_sdk="gradio",
    )
    if len(repo_url.strip()) == 0:
        space_repo_url = f"https://huggingface.co/spaces/{repo_user}/{repo_name}"

    space_path = "/tmp/space"
    os.makedirs(space_path, exist_ok=True)

    logger.info(f"Created repo: {repo_url}")
    logger.info(f"Created space: {space_repo_url}")

    model_repo = utils.clone_hf_repo(
        local_dir=model_path,
        repo_url=repo_url,
        token=huggingface_token,
    )
    model_repo.lfs_track(patterns=LFS_PATTERNS)

    space_repo = utils.clone_hf_repo(
        local_dir=space_path,
        repo_url=space_repo_url,
        token=huggingface_token,
    )
    space_repo.lfs_track(patterns=LFS_PATTERNS)

    # print contents of data_path folder
    logger.info("contents of data_path folder")
    os.system(f"ls -l {data_path}")

    logger.info("processing images")
    concept_prompts = process_images(data_path=data_path, job_config=job_config)
    # convert concept_prompts dict to string
    concept_prompts = ", ".join([f"{k}-> {v}" for k, v in concept_prompts.items()])
    logger.info("done processing images")

    gradient_checkpointing = True if model_name != "multimodalart/sd-fine-tunable" else False
    cache_latents = True if model_name != "multimodalart/sd-fine-tunable" else False

    stop_text_encoder_training = int(job_config.text_encoder_steps_percentage * job_config.num_steps / 100)

    args = TrainingArgs(
        train_text_encoder=True if job_config.text_encoder_steps_percentage > 0 else False,
        stop_text_encoder_training=stop_text_encoder_training,
        pretrained_model_name_or_path=model_name,
        instance_data_dir=os.path.join(data_path, "processed_data"),
        class_data_dir=None,
        output_dir=model_path,
        seed=42,
        resolution=job_config.image_size,
        mixed_precision="fp16",
        train_batch_size=job_config.train_batch_size,
        gradient_accumulation_steps=1,
        use_8bit_adam=True,
        learning_rate=job_config.learning_rate,
        lr_scheduler="polynomial",
        lr_warmup_steps=0,
        max_train_steps=job_config.num_steps,
        gradient_checkpointing=gradient_checkpointing,
        cache_latents=cache_latents,
        revision=None,
    )
    logger.info(args)

    run_training(args)

    co2_consumed = co2_tracker.stop()

    # remove logs folder from model_path
    os.system(f"rm -rf {model_path}/logs")

    model_card = create_model_card(
        dataset_id=data_repo_path,
        model_id=f"{repo_user}/{repo_name}",
        co2=co2_consumed,
    )

    if model_card is not None:
        with open(os.path.join(model_path, "README.md"), "w") as fp:
            fp.write(f"{model_card}")

    logger.info("Pushing model to Hub")
    model_repo.git_pull()
    model_repo.git_add()
    model_repo.git_commit(commit_message="Commit From AutoTrain")
    model_repo.git_push()

    # delete README.md from model_path
    os.system(f"rm -rf {model_path}/README.md")

    # delete .git folder from model_path
    os.system(f"rm -rf {model_path}/.git")

    # copy all contents of model_path to space_path/model
    os.makedirs(os.path.join(space_path, "model"), exist_ok=True)
    os.system(f"cp -r {model_path}/* {space_path}/model")

    # remove old README.md from space_path
    os.system(f"rm -rf {space_path}/README.md")

    # create README.md in space_path
    with open(os.path.join(space_path, "README.md"), "w") as fp:
        fp.write(f"{SPACE_README.format(model_id=repo_name).strip()}")

    # add app.py to space_path
    with open(os.path.join(space_path, "app.py"), "w") as fp:
        fp.write(
            f"{SPACE_APP.format(model_id=repo_name, concept_prompts=concept_prompts, img_size=job_config.image_size).strip()}"
        )

    # add requirements.txt to space_path
    with open(os.path.join(space_path, "requirements.txt"), "w") as fp:
        fp.write(f"{SPACE_REQUIREMENTS.strip()}")

    logger.info("Pushing space to Hub")
    space_repo.git_pull()
    space_repo.git_add()
    space_repo.git_commit(commit_message="Commit From AutoTrain")
    space_repo.git_push()
