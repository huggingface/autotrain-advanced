from itertools import chain

import torch
from datasets import Dataset
from loguru import logger
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


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


class LLMTrainingParams(BaseModel):
    model_name: str = Field("gpt2", title="Model name")
    data_path: str = Field("data", title="Data path")
    train_split: str = Field("train", title="Train data config")
    valid_split: str = Field(None, title="Validation data config")
    text_column: str = Field("text", title="Text column")
    huggingface_token: str = Field(None, title="Huggingface token")
    learning_rate: float = Field(3e-5, title="Learning rate")
    num_train_epochs: int = Field(1, title="Number of training epochs")
    train_batch_size: int = Field(2, title="Training batch size")
    eval_batch_size: int = Field(4, title="Evaluation batch size")
    warmup_ratio: float = Field(0.1, title="Warmup proportion")
    gradient_accumulation_steps: int = Field(1, title="Gradient accumulation steps")
    optimizer: str = Field("adamw_torch", title="Optimizer")
    scheduler: str = Field("linear", title="Scheduler")
    weight_decay: float = Field(0.0, title="Weight decay")
    max_grad_norm: float = Field(1.0, title="Max gradient norm")
    seed: int = Field(42, title="Seed")
    add_eos_token: bool = Field(True, title="Add EOS token")
    block_size: int = Field(-1, title="Block size")
    use_peft: bool = Field(False, title="Use PEFT")
    lora_r: int = Field(16, title="Lora r")
    lora_alpha: int = Field(32, title="Lora alpha")
    lora_dropout: float = Field(0.05, title="Lora dropout")
    training_type: str = Field("generic", title="Training type")
    train_on_inputs: bool = Field(False, title="Train on inputs")
    logging_steps: int = Field(-1, title="Logging steps")
    project_name: str = Field("Project Name", title="Output directory")
    evaluation_strategy: str = Field("epoch", title="Evaluation strategy")
    save_total_limit: int = Field(1, title="Save total limit")
    save_strategy: str = Field("epoch", title="Save strategy")
    auto_find_batch_size: bool = Field(False, title="Auto find batch size")
    fp16: bool = Field(False, title="FP16")
    push_to_hub: bool = Field(False, title="Push to hub")
    use_int8: bool = Field(False, title="Use int8")
    model_max_length: int = Field(1024, title="Model max length")
    repo_id: str = Field(None, title="Repo id")
    use_int4: bool = Field(False, title="Use int4")
    trainer: str = Field("default", title="Trainer type")
    target_modules: str = Field(None, title="Target modules")


def get_target_modules(config):
    if config.target_modules is None:
        return TARGET_MODULES.get(config.model_name)
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
