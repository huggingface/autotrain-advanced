import os
from huggingface_hub import list_models
from autotrain import logger

VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

try:
    XL_MODELS = [
        m.id
        for m in list(
            list_models(
                task="text-to-image",
                sort="downloads",
                limit=200,
                direction=-1,
                filter=["diffusers:StableDiffusionXLPipeline"],
            )
        )
    ]
except Exception:
    logger.info("Unable to reach Hugging Face Hub, using default models as XL models.")
    XL_MODELS = [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/stable-diffusion-xl-base-0.9",
        "diffusers/stable-diffusion-xl-base-1.0",
        "stabilityai/sdxl-turbo",
    ]

def save_model_card_xl(
    repo_id: str,
    base_model: str,
    train_text_encoder: bool,
    instance_prompt: str,
    repo_folder: str = None,
    vae_path: str = None,
):
    img_str = ""
    yaml = f"""---
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
---"""

    model_card = f"""# ModelsLab LoRA DreamBooth Training - {repo_id}
<Gallery />

## Model description
These are {repo_id} LoRA adaption weights for {base_model}.
The weights were trained using [Modelslab](https://modelslab.com).
LoRA for the text encoder was enabled: {train_text_encoder}.
Special VAE used for training: {vae_path}.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)
```py
!pip install -q transformers accelerate peft diffusers
from diffusers import DiffusionPipeline
import torch

pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("{repo_id}", weight_name="pytorch_lora_weights.safetensors", adapter_name="abc")
prompt = "abc of a hacker with a hoodie"
lora_scale = 0.9
image = pipe(
    prompt,
    num_inference_steps=30,
    cross_attention_kwargs={{"scale": lora_scale}},
    generator=torch.manual_seed(0)
).images[0]
image
```

## Trigger words
You should use {instance_prompt} to trigger the image generation.

## Download model
Weights for this model are available in Safetensors format.
[Download]({repo_id}/tree/main) them in the Files & versions tab."""

    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + "\n" + model_card)

def save_model_card(
    repo_id: str,
    base_model: str,
    train_text_encoder: bool,
    instance_prompt: str,
    repo_folder: str = None,
):
    img_str = ""
    yaml = f"""---
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
---"""

    model_description = f"""# ModelsLab LoRA DreamBooth Training - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were trained on {instance_prompt} using [ModelsLab](https://modelslab.com).
LoRA for the text encoder was enabled: {train_text_encoder}.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)
```py
!pip install -q transformers accelerate peft diffusers
from diffusers import DiffusionPipeline
import torch

pipe_id = "Lykon/DreamShaper"
pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("{repo_id}", weight_name="pytorch_lora_weights.safetensors", adapter_name="abc")
prompt = "abc of a hacker with a hoodie"
lora_scale = 0.9
image = pipe(
    prompt,
    num_inference_steps=30,
    cross_attention_kwargs={{"scale": 0.9}},
    generator=torch.manual_seed(0)
).images[0]
image
```"""

    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + "\n" + model_description)