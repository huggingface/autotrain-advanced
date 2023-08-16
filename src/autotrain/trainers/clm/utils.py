from itertools import chain

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from autotrain import logger


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
TARGET_MODULES = {
    "Salesforce/codegen25-7b-multi": "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
}

MODEL_CARD = """
---
tags:
- autotrain
- text-generation
widget:
- text: "I love AutoTrain because "
---

# Model Trained Using AutoTrain
"""


def get_target_modules(config):
    if config.target_modules is None:
        return TARGET_MODULES.get(config.model)
    return config.target_modules.split(",")


def process_data(data, tokenizer, config):
    data = data.to_pandas()
    data = data.fillna("")

    data = data[[config.text_column]]
    if config.add_eos_token:
        data[config.text_column] = data[config.text_column] + tokenizer.eos_token
    data = Dataset.from_pandas(data)
    return data


def group_texts(examples, config):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= config.block_size:
        total_length = (total_length // config.block_size) * config.block_size
    else:
        total_length = 0
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + config.block_size] for i in range(0, total_length, config.block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def tokenize(examples, tokenizer, config):
    output = tokenizer(examples[config.text_column])
    return output


def _tokenize(prompt, tokenizer, config):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding=False,
        return_tensors=None,
    )
    if result["input_ids"][-1] != tokenizer.eos_token_id and config.add_eos_token:
        if len(result["input_ids"]) >= tokenizer.model_max_length:
            result["input_ids"] = result["input_ids"][:-1]
            result["attention_mask"] = result["attention_mask"][:-1]
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def merge_adapter(base_model_path, target_model_path, adapter_path):
    logger.info("Loading adapter...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    model = model.merge_and_unload()

    logger.info("Saving target model...")
    model.save_pretrained(target_model_path)
    tokenizer.save_pretrained(target_model_path)


def create_model_card():
    return MODEL_CARD.strip()
