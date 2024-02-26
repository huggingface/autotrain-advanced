from pathlib import Path

import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms


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
    It pre-processes the images and tokenizes prompts.
    """

    def __init__(self, config, tokenizers, encoder_hidden_states, instance_prompt_encoder_hidden_states):
        self.config = config
        self.tokenizer = tokenizers[0]
        self.size = self.config.resolution
        self.center_crop = self.config.center_crop
        self.tokenizer_max_length = self.config.tokenizer_max_length
        self.instance_data_root = Path(self.config.image_path)
        self.instance_prompt = self.config.prompt
        self.class_data_root = Path(self.config.class_image_path) if self.config.prior_preservation else None
        self.class_prompt = self.config.class_prompt
        self.class_num = self.config.num_class_images

        self.encoder_hidden_states = encoder_hidden_states
        self.instance_prompt_encoder_hidden_states = instance_prompt_encoder_hidden_states

        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(self.instance_data_root).iterdir())

        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        if self.class_data_root is not None:
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if self.class_num is not None:
                self.num_class_images = min(len(self.class_images_path), self.class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size) if self.center_crop else transforms.RandomCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def _tokenize_prompt(self, tokenizer, prompt, tokenizer_max_length=None):
        # this function is here to avoid cyclic import issues
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
                text_inputs = self._tokenize_prompt(
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
                    class_text_inputs = self._tokenize_prompt(
                        self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
                    )
                    example["class_prompt_ids"] = class_text_inputs.input_ids
                    example["class_attention_mask"] = class_text_inputs.attention_mask

        return example


def collate_fn(examples, config):
    pixel_values = [example["instance_images"] for example in examples]

    if not config.xl:
        has_attention_mask = "instance_attention_mask" in examples[0]
        input_ids = [example["instance_prompt_ids"] for example in examples]

        if has_attention_mask:
            attention_mask = [example["instance_attention_mask"] for example in examples]

    if config.prior_preservation:
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
