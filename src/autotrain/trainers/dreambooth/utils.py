import os


VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
XL_MODELS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-xl-base-0.9",
    "diffusers/stable-diffusion-xl-base-1.0",
    "stabilityai/sdxl-turbo",
]


def save_model_card_xl(
    repo_id: str,
    base_model=str,
    train_text_encoder=False,
    instance_prompt=str,
    repo_folder=None,
    vae_path=None,
):
    img_str = ""
    yaml = f"""
---
tags:
- autotrain
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- text-to-image
- diffusers
- lora
- template:sd-lora
{img_str}
base_model: {base_model}
instance_prompt: {instance_prompt}
license: openrail++
---
    """

    model_card = f"""
# AutoTrain SDXL LoRA DreamBooth - {repo_id}

<Gallery />

## Model description

These are {repo_id} LoRA adaption weights for {base_model}.

The weights were trained  using [DreamBooth](https://dreambooth.github.io/).

LoRA for the text encoder was enabled: {train_text_encoder}.

Special VAE used for training: {vae_path}.

## Trigger words

You should use {instance_prompt} to trigger the image generation.

## Download model

Weights for this model are available in Safetensors format.

[Download]({repo_id}/tree/main) them in the Files & versions tab.

"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def save_model_card(
    repo_id: str,
    base_model=str,
    train_text_encoder=False,
    instance_prompt=str,
    repo_folder=None,
):
    img_str = ""
    model_description = f"""
# AutoTrain LoRA DreamBooth - {repo_id}

These are LoRA adaption weights for {base_model}. The weights were trained on {instance_prompt} using [DreamBooth](https://dreambooth.github.io/).
LoRA for the text encoder was enabled: {train_text_encoder}.
"""

    yaml = f"""
---
tags:
- autotrain
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
- template:sd-lora
{img_str}
base_model: {base_model}
instance_prompt: {instance_prompt}
license: openrail++
---
    """
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_description)
