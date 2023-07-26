import hashlib
import itertools
import os
from pathlib import Path
from typing import Dict

import torch
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from loguru import logger
from packaging import version
from tqdm import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from autotrain.trainers.dreambooth.datasets import PromptDataset


VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
XL_MODELS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-xl-base-0.9",
    "diffusers/stable-diffusion-xl-base-1.0",
]


def create_model_card(repo_id: str, base_model: str, train_text_encoder: bool, prompt: str, repo_folder: str):
    if train_text_encoder:
        text_encoder_text = "trained"
    else:
        text_encoder_text = "not trained"
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
    model_card = f"""
# DreamBooth trained by AutoTrain

Test enoder was {text_encoder_text}.

"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


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
                config.model,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=config.revision,
            )
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                config.model,
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


def load_model_components(config, device, weight_dtype):
    tokenizers = []
    tokenizers.append(
        AutoTokenizer.from_pretrained(
            config.model,
            subfolder="tokenizer",
            revision=config.revision,
            use_fast=False,
        )
    )
    if config.xl:
        tokenizers.append(
            AutoTokenizer.from_pretrained(
                config.model,
                subfolder="tokenizer_2",
                revision=config.revision,
                use_fast=False,
            )
        )

    cls_text_encoders = []
    cls_text_encoders.append(
        import_model_class_from_model_name_or_path(config.model, config.revision),
    )
    if config.xl:
        cls_text_encoders.append(
            import_model_class_from_model_name_or_path(config.model, config.revision, subfolder="text_encoder_2")
        )

    text_encoders = []
    text_encoders.append(
        cls_text_encoders[0].from_pretrained(
            config.model,
            subfolder="text_encoder",
            revision=config.revision,
        )
    )
    if config.xl:
        text_encoders.append(
            cls_text_encoders[1].from_pretrained(
                config.model,
                subfolder="text_encoder_2",
                revision=config.revision,
            )
        )

    try:
        vae = AutoencoderKL.from_pretrained(config.model, subfolder="vae", revision=config.revision)
    except OSError:
        logger.warning("No VAE found. Training without VAE.")
        vae = None

    unet = UNet2DConditionModel.from_pretrained(
        config.model,
        subfolder="unet",
        revision=config.revision,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(config.model, subfolder="scheduler")

    # TODO: non-peft version
    if vae is not None:
        vae.requires_grad_(False)
    for _text_encoder in text_encoders:
        _text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if vae is not None:
        if config.xl:
            vae.to(device, dtype=torch.float32)
        else:
            vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    for _text_encoder in text_encoders:
        _text_encoder.to(device, dtype=weight_dtype)

    return tokenizers, text_encoders, vae, unet, noise_scheduler


def enable_xformers(unet, config):
    if config.xformers:
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
        lr=config.lr,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )
    return optimizer


def pre_compute_text_embeddings(config, tokenizers, text_encoders):
    if config.pre_compute_text_embeddings:
        tokenizer = tokenizers[0]
        text_encoder = text_encoders[0]

        def compute_text_embeddings(prompt):
            with torch.no_grad():
                text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=config.tokenizer_max_length)
                prompt_embeds = encode_prompt(
                    text_encoder,
                    text_inputs.input_ids,
                    text_inputs.attention_mask,
                    text_encoder_use_attention_mask=config.text_encoder_use_attention_mask,
                )

            return prompt_embeds

        pre_computed_encoder_hidden_states = compute_text_embeddings(config.prompt)

        # disable validation prompt for now
        # validation_prompt_negative_prompt_embeds = compute_text_embeddings("")

        # if args.validation_prompt is not None:
        #     validation_prompt_encoder_hidden_states = compute_text_embeddings(args.validation_prompt)
        # else:
        #     validation_prompt_encoder_hidden_states = None

        if config.prompt is not None:
            pre_computed_instance_prompt_encoder_hidden_states = compute_text_embeddings(config.prompt)
        else:
            pre_computed_instance_prompt_encoder_hidden_states = None

    else:
        pre_computed_encoder_hidden_states = None
        # validation_prompt_encoder_hidden_states = None
        pre_computed_instance_prompt_encoder_hidden_states = None

    return pre_computed_encoder_hidden_states, pre_computed_instance_prompt_encoder_hidden_states
