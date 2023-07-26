import json
import os

import diffusers
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import StableDiffusionXLPipeline
from diffusers.loaders import LoraLoaderMixin, text_encoder_lora_state_dict
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from loguru import logger
from PIL import Image

from autotrain import utils as at_utils
from autotrain.params import DreamboothParams
from autotrain.trainers.dreambooth import utils
from autotrain.trainers.dreambooth.datasets import DreamBoothDataset, collate_fn
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.dreambooth.trainer import Trainer


def train(config):
    if isinstance(config, dict):
        config = DreamBoothTrainingParams(**config)
    config.prompt = str(config.prompt).strip()
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output, logging_dir=os.path.join(config.output, "logs")
    )

    if config.fp16:
        mixed_precision = "fp16"
    elif config.bf16:
        mixed_precision = "bf16"
    else:
        mixed_precision = "no"

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation,
        mixed_precision=mixed_precision,
        log_with="tensorboard" if config.logging else None,
        project_config=accelerator_project_config,
    )

    if config.train_text_encoder and config.gradient_accumulation > 1 and accelerator.num_processes > 1:
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
    if config.prior_preservation:
        utils.setup_prior_preservation(accelerator, config)

    # Handle the repository creation
    if accelerator.is_main_process:
        if config.output is not None:
            os.makedirs(config.output, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    tokenizers, text_encoders, vae, unet, noise_scheduler = utils.load_model_components(
        config, accelerator.device, weight_dtype
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
                safe_serialization=True,
            )
        elif len(text_encoder_lora_layers_to_save) == 1:
            LoraLoaderMixin.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_lora_layers_to_save[0],
                safe_serialization=True,
            )
        elif len(text_encoder_lora_layers_to_save) == 2:
            StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_lora_layers_to_save[0],
                text_encoder_2_lora_layers=text_encoder_lora_layers_to_save[1],
                safe_serialization=True,
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
        config.lr = config.lr * config.gradient_accumulation * config.batch_size * accelerator.num_processes

    optimizer = utils.get_optimizer(config, unet_lora_parameters, text_lora_parameters)

    encoder_hs, instance_prompt_encoder_hs = utils.pre_compute_text_embeddings(
        config=config, text_encoders=text_encoders, tokenizers=tokenizers
    )
    train_dataset = DreamBoothDataset(
        config=config,
        tokenizers=tokenizers,
        encoder_hidden_states=encoder_hs,
        instance_prompt_encoder_hidden_states=instance_prompt_encoder_hs,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, config),
        num_workers=config.dataloader_num_workers,
    )
    trainer = Trainer(
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

    if config.push_to_hub:
        trainer.push_to_hub()


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


@at_utils.job_watcher
def train_ui(co2_tracker, payload, huggingface_token, model_path):
    data_repo_path = f"{payload['username']}/autotrain-data-{payload['proj_name']}"
    data_path = "/tmp/data"
    data_repo = at_utils.clone_hf_repo(
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
    logger.info(f"job_config: {job_config}")
    job_config = DreamboothParams(**job_config)
    logger.info(f"job_config: {job_config}")

    logger.info("Create model repo")
    project_name = payload["proj_name"]
    repo_name = f"autotrain-{project_name}"
    repo_user = payload["username"]

    # print contents of data_path folder
    logger.info("contents of data_path folder")
    os.system(f"ls -l {data_path}")

    logger.info("processing images")
    concept_prompts = process_images(data_path=data_path, job_config=job_config)
    # convert concept_prompts dict to string
    concept_prompts = list(concept_prompts.values())[0]
    logger.info("done processing images")

    xl = False
    if model_name in utils.XL_MODELS:
        xl = True

    args = DreamBoothTrainingParams(
        model=model_name,
        image_path=os.path.join(data_path, "processed_data"),
        output=model_path,
        seed=42,
        resolution=job_config.image_size,
        fp16=True,
        batch_size=job_config.train_batch_size,
        gradient_accumulation=job_config.gradient_accumulation_steps,
        use_8bit_adam=True,
        lr=job_config.learning_rate,
        scheduler="constant",
        warmup_steps=0,
        num_steps=job_config.num_steps,
        revision=None,
        push_to_hub=True,
        hub_model_id=f"{repo_user}/{repo_name}",
        xl=xl,
        prompt=concept_prompts,
        hub_token=huggingface_token,
    )
    train(args)

    _ = co2_tracker.stop()
