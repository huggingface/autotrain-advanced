import ast
import os
from itertools import chain

import requests
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from autotrain import logger


DEFAULT_CHAT_TEMPLATE = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


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
license: other
---

# Model Trained Using AutoTrain

This model was trained using AutoTrain. For more information, please visit [AutoTrain](https://hf.co/docs/autotrain).

# Usage

```python

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "PATH_TO_THIS_REPO"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()

# Prompt content: "hi"
messages = [
    {"role": "user", "content": "hi"}
]

input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
output_ids = model.generate(input_ids.to('cuda'))
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

# Model response: "Hello! How can I assist you today?"
print(response)
```

"""

PEFT_HANDLER = """
from typing import  Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from peft import PeftModel
import json
import os


class EndpointHandler():
    def __init__(self, path=""):
        base_model_path = json.load(open(os.path.join(path, "training_params.json")))["model"]
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, path)
        model = model.merge_and_unload()
        self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def __call__(self, data: Any) -> List[List[Dict[str, float]]]:
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None)
        if parameters is not None:
            prediction = self.pipeline(inputs, **parameters)
        else:
            prediction = self.pipeline(inputs)
        return prediction

"""

REQUIREMENTS_TXT = """
peft==0.9.0
transformers==4.38.2
"""


def preprocess_reward(examples, tokenizer):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen, truncation=True)
        tokenized_rejected = tokenizer(rejected, truncation=True)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples


def get_target_modules(config):
    if config.target_modules is None:
        return TARGET_MODULES.get(config.model)
    if config.target_modules.strip() == "":
        return TARGET_MODULES.get(config.model)
    if config.target_modules.strip().lower() == "all-linear":
        return "all-linear"
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

    tokenizer = AutoTokenizer.from_pretrained(
        target_model_path,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    logger.info("Saving target model...")
    model.save_pretrained(target_model_path)
    tokenizer.save_pretrained(target_model_path)


def create_model_card():
    return MODEL_CARD.strip()


def pause_endpoint(params):
    endpoint_id = os.environ["ENDPOINT_ID"]
    username = endpoint_id.split("/")[0]
    project_name = endpoint_id.split("/")[1]
    api_url = f"https://api.endpoints.huggingface.cloud/v2/endpoint/{username}/{project_name}/pause"
    headers = {"Authorization": f"Bearer {params.token}"}
    r = requests.post(api_url, headers=headers, timeout=30)
    return r.json()


def create_peft_handler(config):
    txt = PEFT_HANDLER.strip()
    with open(f"{config.project_name}/handler.py", "w", encoding="utf-8") as f:
        f.write(txt)


def create_requirements_txt(config):
    txt = REQUIREMENTS_TXT.strip()
    with open(f"{config.project_name}/requirements.txt", "w", encoding="utf-8") as f:
        f.write(txt)


def apply_chat_template(
    example,
    tokenizer,
    config,
):
    # kudos to Hugging Face H4 Team for this snippet
    if config.trainer in ("default", "sft"):
        messages = example[config.text_column]
        if isinstance(messages, str):
            messages = ast.literal_eval(messages)
        example[config.text_column] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    elif config.trainer == "reward":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            if isinstance(chosen_messages, str):
                chosen_messages = ast.literal_eval(chosen_messages)
            if isinstance(rejected_messages, str):
                rejected_messages = ast.literal_eval(rejected_messages)
            example["chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif config.trainer == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            # For DPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            if isinstance(example["chosen"], str):
                example["chosen"] = ast.literal_eval(example["chosen"])
            if isinstance(example["rejected"], str):
                example["rejected"] = ast.literal_eval(example["rejected"])
            prompt_messages = example["chosen"][:-1]
            chosen_messages = example["chosen"][-1:]
            rejected_messages = example["rejected"][-1:]
            example["chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            example["prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
    else:
        raise ValueError(
            f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
        )
    return example
