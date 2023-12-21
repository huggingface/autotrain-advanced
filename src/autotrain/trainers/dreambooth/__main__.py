import argparse
import json
import os

import diffusers
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import StableDiffusionXLPipeline
from diffusers.loaders import LoraLoaderMixin, text_encoder_lora_state_dict
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from diffusers.models.lora import LoRALinearLayer
from huggingface_hub import snapshot_download

from autotrain.trainers.common import monitor, pause_space, remove_autotrain_data
from autotrain.trainers.dreambooth import utils
from autotrain.trainers.dreambooth.datasets import DreamBoothDataset, collate_fn
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.dreambooth.trainer import Trainer


def parse_args():
    # get training_config.json from the end user
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()


@monitor
def train(config):
    if isinstance(config, dict):
        config = DreamBoothTrainingParams(**config)
    config.prompt = str(config.prompt).strip()

    if config.model in utils.XL_MODELS:
        config.xl = True

    if config.repo_id is None and config.username is not None:
        config.repo_id = f"{config.username}/{config.project_name}"

    if config.project_name == "/tmp/model":
        snapshot_download(
            repo_id=config.image_path,
            local_dir=config.project_name,
            token=config.token,
            repo_type="dataset",
        )
        config.image_path = "/tmp/model/concept1/"
    if config.image_path == f"{config.project_name}/autotrain-data":
        config.image_path = os.path.join(config.image_path, "concept1")

    accelerator_project_config = ProjectConfiguration(
        project_dir=config.project_name,
        logging_dir=os.path.join(config.project_name, "logs"),
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
        if config.project_name is not None:
            os.makedirs(config.project_name, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    tokenizers, text_encoders, vae, unet, noise_scheduler = utils.load_model_components(
        config, accelerator.device, weight_dtype
    )

    utils.enable_xformers(unet, config)
    utils.enable_gradient_checkpointing(unet, text_encoders, config)

    unet_lora_parameters = []
    for attn_processor_name, attn_processor in unet.attn_processors.items():
        # Parse the attention module.
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)

        # Set the `lora_layer` attribute of the attention-related matrices.
        attn_module.to_q.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_q.in_features,
                out_features=attn_module.to_q.out_features,
            )
        )
        attn_module.to_k.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
            )
        )
        attn_module.to_v.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_v.in_features,
                out_features=attn_module.to_v.out_features,
            )
        )
        attn_module.to_out[0].set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
            )
        )

        # Accumulate the LoRA params to optimize.
        unet_lora_parameters.extend(attn_module.to_q.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_k.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_v.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_out[0].lora_layer.parameters())

        if not config.xl:
            if isinstance(
                attn_processor,
                (
                    AttnAddedKVProcessor,
                    SlicedAttnAddedKVProcessor,
                    AttnAddedKVProcessor2_0,
                ),
            ):
                attn_module.add_k_proj.set_lora_layer(
                    LoRALinearLayer(
                        in_features=attn_module.add_k_proj.in_features,
                        out_features=attn_module.add_k_proj.out_features,
                    )
                )
                attn_module.add_v_proj.set_lora_layer(
                    LoRALinearLayer(
                        in_features=attn_module.add_v_proj.in_features,
                        out_features=attn_module.add_v_proj.out_features,
                    )
                )
                unet_lora_parameters.extend(attn_module.add_k_proj.lora_layer.parameters())
                unet_lora_parameters.extend(attn_module.add_v_proj.lora_layer.parameters())

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

        if config.xl:
            if len(text_encoder_lora_layers_to_save) > 1:
                StableDiffusionXLPipeline.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                    text_encoder_lora_layers=text_encoder_lora_layers_to_save[0],
                    text_encoder_2_lora_layers=text_encoder_lora_layers_to_save[1],
                    safe_serialization=True,
                )
            else:
                StableDiffusionXLPipeline.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                    text_encoder_lora_layers=None,
                    text_encoder_2_lora_layers=None,
                    safe_serialization=True,
                )
        else:
            if len(text_encoder_lora_layers_to_save) > 0:
                LoraLoaderMixin.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                    text_encoder_lora_layers=text_encoder_lora_layers_to_save[0],
                    safe_serialization=True,
                )
            else:
                LoraLoaderMixin.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                    text_encoder_lora_layers=None,
                    safe_serialization=True,
                )

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

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
        LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alpha=network_alphas, unet=unet_)

        if config.xl:
            text_encoder_state_dict = {k: v for k, v in lora_state_dict.items() if "text_encoder." in k}
            LoraLoaderMixin.load_lora_into_text_encoder(
                text_encoder_state_dict,
                network_alphas=network_alphas,
                text_encoder=text_encoders_[0],
            )

            text_encoder_2_state_dict = {k: v for k, v in lora_state_dict.items() if "text_encoder_2." in k}
            LoraLoaderMixin.load_lora_into_text_encoder(
                text_encoder_2_state_dict,
                network_alphas=network_alphas,
                text_encoder=text_encoders_[1],
            )
        else:
            LoraLoaderMixin.load_lora_into_text_encoder(
                lora_state_dict,
                network_alphas=network_alphas,
                text_encoder=text_encoders_[0],
            )

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

    # remove token key from training_params.json located in output directory
    # first check if file exists
    if os.path.exists(f"{config.project_name}/training_params.json"):
        training_params = json.load(open(f"{config.project_name}/training_params.json"))
        if "token" in training_params:
            training_params.pop("token")
            json.dump(
                training_params,
                open(f"{config.project_name}/training_params.json", "w"),
            )

    # add config.prompt as a text file in the output directory
    with open(f"{config.project_name}/prompt.txt", "w") as f:
        f.write(config.prompt)

    if config.push_to_hub:
        remove_autotrain_data(config)
        trainer.push_to_hub()

    pause_space(config)


if __name__ == "__main__":
    args = parse_args()
    training_config = json.load(open(args.training_config))
    config = DreamBoothTrainingParams(**training_config)
    train(config)
