import gc
import hashlib
import itertools
import math
import os
import shutil
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin, text_encoder_lora_state_dict
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from loguru import logger
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer, PretrainedConfig


XL_MODELS = ["stabilityai/stable-diffusion-xl-base-0.9"]


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


class DreamBoothDatasetXL(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        instance_data_root,
        class_data_root=None,
        class_num=None,
        size=1024,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop

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
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

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
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

        return example


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        config,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        instance_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.config = config
        self.encoder_hidden_states = encoder_hidden_states
        self.instance_prompt_encoder_hidden_states = instance_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

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
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if not self.config.xl:
            if self.encoder_hidden_states is not None:
                example["instance_prompt_ids"] = self.encoder_hidden_states
            else:
                text_inputs = tokenize_prompt(
                    self.tokenizer, self.instance_prompt, tokenizer_max_length=self.tokenizer_max_length
                )
                example["instance_prompt_ids"] = text_inputs.input_ids
                example["instance_attention_mask"] = text_inputs.attention_mask

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            if not self.config.xl:
                if self.instance_prompt_encoder_hidden_states is not None:
                    example["class_prompt_ids"] = self.instance_prompt_encoder_hidden_states
                else:
                    class_text_inputs = tokenize_prompt(
                        self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
                    )
                    example["class_prompt_ids"] = class_text_inputs.input_ids
                    example["class_attention_mask"] = class_text_inputs.attention_mask

        return example


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def collate_fn(examples, config):
    pixel_values = [example["instance_images"] for example in examples]

    if not config.xl:
        has_attention_mask = "instance_attention_mask" in examples[0]
        input_ids = [example["instance_prompt_ids"] for example in examples]

        if has_attention_mask:
            attention_mask = [example["instance_attention_mask"] for example in examples]

    if config.with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        if not config.xl:
            input_ids += [example["class_prompt_ids"] for example in examples]
            if has_attention_mask:
                attention_mask += [example["class_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "pixel_values": pixel_values,
    }

    if not config.xl:
        input_ids = torch.cat(input_ids, dim=0)
        batch["input_ids"] = input_ids
        if has_attention_mask:
            # attention_mask = torch.cat(attention_mask, dim=0)
            batch["attention_mask"] = attention_mask

    return batch


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def encode_prompt_xl(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []
    # logger.info(f"Computing text embeddings for prompt: {prompt}")
    # logger.info(f"Text encoders: {text_encoders}")
    # logger.info(f"Tokenizers: {tokenizers}")

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt).input_ids
            # logger.info(f"Text input ids: {text_input_ids}")
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def unet_attn_processors_state_dict(unet) -> Dict[str, torch.tensor]:
    r"""
    Returns:
        a state dict containing just the attention processor parameters.
    """
    attn_processors = unet.attn_processors

    attn_processors_state_dict = {}

    for attn_processor_key, attn_processor in attn_processors.items():
        for parameter_key, parameter in attn_processor.state_dict().items():
            attn_processors_state_dict[f"{attn_processor_key}.{parameter_key}"] = parameter

    return attn_processors_state_dict


def save_model_card(repo_id: str, base_model=str, prompt=str, repo_folder=None):
    yaml = f"""
---
base_model: {base_model}
instance_prompt: {prompt}
tags:
- text-to-image
- diffusers
- autotrain
inference: true
---
    """
    model_card = """
# DreamBooth trained by AutoTrain

"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def setup_prior_preservation(accelerator, config):
    class_images_dir = Path(config.class_image_path)
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
        if config.xl:
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                config.model_name,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=config.revision,
            )
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                config.model_name,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=config.revision,
            )
        pipeline.set_progress_bar_config(disable=True)

        num_new_images = config.num_class_images - cur_class_images
        logger.info(f"Number of class images to sample: {num_new_images}.")

        sample_dataset = PromptDataset(config.class_prompt, num_new_images)
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


def load_model_components(config, accelerator):
    tokenizers = []
    tokenizers.append(
        AutoTokenizer.from_pretrained(
            config.model_name,
            subfolder="tokenizer",
            revision=config.revision,
            use_fast=False,
        )
    )
    if config.xl:
        tokenizers.append(
            AutoTokenizer.from_pretrained(
                config.model_name,
                subfolder="tokenizer_2",
                revision=config.revision,
                use_fast=False,
            )
        )

    cls_text_encoders = []
    cls_text_encoders.append(
        import_model_class_from_model_name_or_path(config.model_name, config.revision),
    )
    if config.xl:
        cls_text_encoders.append(
            import_model_class_from_model_name_or_path(config.model_name, config.revision, subfolder="text_encoder_2")
        )

    text_encoders = []
    text_encoders.append(
        cls_text_encoders[0].from_pretrained(
            config.model_name,
            subfolder="text_encoder",
            revision=config.revision,
        )
    )
    if config.xl:
        text_encoders.append(
            cls_text_encoders[1].from_pretrained(
                config.model_name,
                subfolder="text_encoder_2",
                revision=config.revision,
            )
        )

    try:
        vae = AutoencoderKL.from_pretrained(config.model_name, subfolder="vae", revision=config.revision)
    except OSError:
        logger.warning("No VAE found. Training without VAE.")
        vae = None

    unet = UNet2DConditionModel.from_pretrained(
        config.model_name,
        subfolder="unet",
        revision=config.revision,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(config.model_name, subfolder="scheduler")

    # TODO: non-peft version
    if vae is not None:
        vae.requires_grad_(False)
    for _text_encoder in text_encoders:
        _text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    weight_dtype = torch.float16
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16

    if vae is not None:
        if config.xl:
            vae.to(accelerator.device, dtype=torch.float32)  # TODO: sdxl vae type float32?
        else:
            vae.to(accelerator.device, dtype=weight_dtype)  # TODO: sdxl vae type float32?
    unet.to(accelerator.device, dtype=weight_dtype)
    for _text_encoder in text_encoders:
        _text_encoder.to(accelerator.device, dtype=weight_dtype)

    return tokenizers, text_encoders, vae, unet, noise_scheduler, weight_dtype


def enable_xformers(unet, config):
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


def get_optimizer(config, unet_lora_parameters, text_lora_parameters):
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

    if len(text_lora_parameters) == 0:
        params_to_optimize = unet_lora_parameters
    elif len(text_lora_parameters) == 1:
        params_to_optimize = itertools.chain(unet_lora_parameters, text_lora_parameters[0])
    elif len(text_lora_parameters) == 2:
        params_to_optimize = itertools.chain(unet_lora_parameters, text_lora_parameters[0], text_lora_parameters[1])
    else:
        raise ValueError("More than 2 text encoders are not supported.")

    optimizer = optimizer_class(
        params_to_optimize,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )
    return optimizer


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
        self.num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
        if self.config.max_train_steps is None:
            self.config.max_train_steps = self.config.num_train_epochs * self.num_update_steps_per_epoch
            overrode_max_train_steps = True

        self.scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.config.max_train_steps * self.accelerator.num_processes,
            num_cycles=self.config.lr_num_cycles,
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
                    self.unet, self.text_encoder1, self.optimizer, self.train_dataloader, self.scheduler
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
            self.unet, self.optimizer, self.train_dataloader, self.scheduler = accelerator.prepare(
                self.unet, self.optimizer, self.train_dataloader, self.scheduler
            )

        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )
        if overrode_max_train_steps:
            self.config.max_train_steps = self.config.num_train_epochs * self.num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.config.num_train_epochs = math.ceil(self.config.max_train_steps / self.num_update_steps_per_epoch)

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("dreambooth")

        self.total_batch_size = (
            self.config.train_batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps
        )
        logger.info(self.config)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(self.train_dataloader)}")
        logger.info(f"  Num Epochs = {self.config.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.config.max_train_steps}")
        self.global_step = 0
        self.first_epoch = 0

        if config.resume_from_checkpoint:
            self._resume_from_checkpoint()

    def compute_text_embeddings(self, prompt):
        logger.info(f"Computing text embeddings for prompt: {prompt}")
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt_xl(self.text_encoders, self.tokenizers, prompt)
            prompt_embeds = prompt_embeds.to(self.accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(self.accelerator.device)
        return prompt_embeds, pooled_prompt_embeds

    def _setup_xl(self):
        def compute_time_ids():
            # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
            original_size = (self.config.resolution, self.config.resolution)
            target_size = (self.config.resolution, self.config.resolution)
            # crops_coords_top_left = (self.config.crops_coords_top_left_h, self.config.crops_coords_top_left_w)
            crops_coords_top_left = (0, 0)
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids])
            add_time_ids = add_time_ids.to(self.accelerator.device, dtype=self.weight_dtype)
            return add_time_ids

        # Handle instance prompt.
        instance_time_ids = compute_time_ids()
        if not self.config.train_text_encoder:
            instance_prompt_hidden_states, instance_pooled_prompt_embeds = self.compute_text_embeddings(
                self.config.instance_prompt
            )

        # Handle class prompt for prior-preservation.
        if self.config.with_prior_preservation:
            class_time_ids = compute_time_ids()
            if not self.config.train_text_encoder:
                class_prompt_hidden_states, class_pooled_prompt_embeds = self.compute_text_embeddings(
                    self.config.class_prompt
                )

        if not self.config.train_text_encoder:
            gc.collect()
            torch.cuda.empty_cache()

        self.add_time_ids = instance_time_ids
        if self.config.with_prior_preservation:
            self.add_time_ids = torch.cat([self.add_time_ids, class_time_ids], dim=0)

        if not self.config.train_text_encoder:
            self.prompt_embeds = instance_prompt_hidden_states
            self.unet_add_text_embeds = instance_pooled_prompt_embeds
            if self.config.with_prior_preservation:
                self.prompt_embeds = torch.cat([self.prompt_embeds, class_prompt_hidden_states], dim=0)
                self.unet_add_text_embeds = torch.cat([self.unet_add_text_embeds, class_pooled_prompt_embeds], dim=0)
        else:
            self.tokens_one = tokenize_prompt(self.tokenizers[0], self.config.instance_prompt).input_ids
            self.tokens_two = tokenize_prompt(self.tokenizers[1], self.config.instance_prompt).input_ids
            if self.config.with_prior_preservation:
                class_tokens_one = tokenize_prompt(self.tokenizers[0], self.config.class_prompt).input_ids
                class_tokens_two = tokenize_prompt(self.tokenizers[1], self.config.class_prompt).input_ids
                self.tokens_one = torch.cat([self.tokens_one, class_tokens_one], dim=0)
                self.tokens_two = torch.cat([self.tokens_two, class_tokens_two], dim=0)

        ### XL specific stuff ends

    def _resume_from_checkpoint(self):
        if self.config.resume_from_checkpoint != "latest":
            path = os.path.basename(self.config.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(self.config.output_dir)
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
            self.accelerator.load_state(os.path.join(self.config.output_dir, path))
            self.global_step = int(path.split("-")[1])

            resume_global_step = self.global_step * self.config.gradient_accumulation_steps
            self.first_epoch = self.global_step // self.num_update_steps_per_epoch
            self.resume_step = resume_global_step % (
                self.num_update_steps_per_epoch * self.config.gradient_accumulation_steps
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

        if self.config.with_prior_preservation:
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
        # logger.info("Clipping gradients")
        # logger.info(len(self.text_lora_parameters))
        if self.accelerator.sync_gradients:
            if len(self.text_lora_parameters) == 0:
                params_to_clip = self.unet_lora_parameters
            elif len(self.text_lora_parameters) == 1:
                params_to_clip = itertools.chain(self.unet_lora_parameters, self.text_lora_parameters[0])
            elif len(self.text_lora_parameters) == 2:
                params_to_clip = itertools.chain(
                    self.unet_lora_parameters, self.text_lora_parameters[0], self.text_lora_parameters[1]
                )
            else:
                raise ValueError("More than 2 text encoders are not supported.")
            self.accelerator.clip_grad_norm_(params_to_clip, self.config.max_grad_norm)

    def _save_checkpoint(self):
        if self.accelerator.is_main_process:
            if self.global_step % self.config.checkpointing_steps == 0:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if self.config.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(self.config.output_dir)
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
                            removing_checkpoint = os.path.join(self.config.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
                self.accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

    def _get_model_pred(self, batch, channels, noisy_model_input, timesteps, bsz):
        if self.config.xl:
            if not self.config.train_text_encoder:
                unet_added_conditions = {
                    "time_ids": self.add_time_ids.repeat(bsz, 1),
                    "text_embeds": self.unet_add_text_embeds.repeat(bsz, 1),
                }
                model_pred = self.unet(
                    noisy_model_input,
                    timesteps,
                    self.prompt_embeds.repeat(bsz, 1, 1),
                    added_cond_kwargs=unet_added_conditions,
                ).sample
                # logger.info(noisy_model_input)
                # logger.info(timesteps)
                # logger.info(self.add_time_ids)
                # logger.info(self.prompt_embeds)
                # logger.info(self.unet_add_text_embeds)
                # logger.info(unet_added_conditions)
            else:
                unet_added_conditions = {"time_ids": self.add_time_ids.repeat(bsz, 1)}
                prompt_embeds, pooled_prompt_embeds = encode_prompt_xl(
                    text_encoders=self.text_encoders,
                    tokenizers=None,
                    prompt=None,
                    text_input_ids_list=[self.tokens_one, self.tokens_two],
                )
                unet_added_conditions.update({"text_embeds": pooled_prompt_embeds.repeat(bsz, 1)})
                prompt_embeds = prompt_embeds.repeat(bsz, 1, 1)
                model_pred = self.unet(
                    noisy_model_input, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions
                ).sample

        else:
            if self.config.pre_compute_text_embeddings:
                encoder_hidden_states = batch["input_ids"]
            else:
                encoder_hidden_states = encode_prompt(
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
                noisy_model_input, timesteps, encoder_hidden_states, class_labels=class_labels
            ).sample

        return model_pred

    def train(self):
        progress_bar = tqdm(
            range(self.global_step, self.config.max_train_steps), disable=not self.accelerator.is_local_main_process
        )
        progress_bar.set_description("Steps")

        for epoch in range(self.first_epoch, self.config.num_train_epochs):
            self.unet.train()

            if self.config.train_text_encoder:
                self.text_encoder1.train()
                if self.config.xl:
                    self.text_encoder2.train()

            for step, batch in enumerate(self.train_dataloader):
                # Skip steps until we reach the resumed step
                if self.config.resume_from_checkpoint and epoch == self.first_epoch and step < self.resume_step:
                    if step % self.config.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with self.accelerator.accumulate(self.unet):
                    if self.config.xl:
                        pixel_values = batch["pixel_values"]
                    else:
                        pixel_values = batch["pixel_values"].to(dtype=self.weight_dtype)

                    # logger.info(pixel_values)
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
                        0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
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

                logs = {"loss": loss.detach().item(), "lr": self.scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=self.global_step)

                if self.global_step >= self.config.max_train_steps:
                    break

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.unet = self.accelerator.unwrap_model(self.unet)
            self.unet = self.unet.to(torch.float32)
            unet_lora_layers = unet_attn_processors_state_dict(self.unet)
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
                    save_directory=self.config.output_dir,
                    unet_lora_layers=unet_lora_layers,
                    text_encoder_lora_layers=text_encoder_lora_layers_1,
                    text_encoder_2_lora_layers=text_encoder_lora_layers_2,
                )
            else:
                LoraLoaderMixin.save_lora_weights(
                    save_directory=self.config.output_dir,
                    unet_lora_layers=unet_lora_layers,
                    text_encoder_lora_layers=text_encoder_lora_layers_1,
                )
        self.accelerator.end_training()
