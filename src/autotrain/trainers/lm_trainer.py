import os
from itertools import chain

import torch
from datasets import Dataset, load_dataset
from loguru import logger
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from autotrain import utils
from autotrain.params import LMTrainingParams


TEXT_COLUMN = "autotrain_text"

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

EVAL_METRICS = ("eval_loss",)

MODEL_CARD = """
---
tags:
- autotrain
- text-generation
widget:
- text: "I love AutoTrain because "
datasets:
- {dataset}
co2_eq_emissions:
  emissions: {co2}
---

# Model Trained Using AutoTrain

- Problem type: Text Generation
- CO2 Emissions (in grams): {co2:.4f}

## Validation Metrics
{validation_metrics}
"""

HANDLER_CONTENT = """
from typing import Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch


class EndpointHandler:
    def __init__(self, path=""):
        # load model and processor from path
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, load_in_8bit=True, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.eval()

    def __call__(self, data: Dict[str, Any]) -> Dict[str, str]:
        '''
        Args:
            data (:dict:):
                The payload with the text prompt and generation parameters.
        '''
        # process input
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None)

        # preprocess
        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids

        # pass inputs with all kwargs in data
        if parameters is not None:
            outputs = self.model.generate(input_ids=input_ids, **parameters)
        else:
            outputs = self.model.generate(input_ids=input_ids)

        # postprocess the prediction
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return [{"generated_text": prediction}]
"""

HANDLER_CONTENT_PEFT = """
from typing import Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch


class EndpointHandler:
    def __init__(self, path=""):
        # load model and processor from path
        config = PeftConfig.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, torch_dtype=torch.float16, load_in_8bit=True, device_map="auto"
        )
        self.model = PeftModel.from_pretrained(model, path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        self.model.eval()

    def __call__(self, data: Dict[str, Any]) -> Dict[str, str]:
        '''
        Args:
            data (:dict:):
                The payload with the text prompt and generation parameters.
        '''
        # process input
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None)

        # preprocess
        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids

        # pass inputs with all kwargs in data
        if parameters is not None:
            outputs = self.model.generate(input_ids=input_ids, **parameters)
        else:
            outputs = self.model.generate(input_ids=input_ids)

        # postprocess the prediction
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return [{"generated_text": prediction}]
"""


REQUIREMENTS = """
accelerate==0.18.0
transformers==4.28.1
git+https://github.com/huggingface/peft.git
bitsandbytes
tokenizers>=0.13.3
"""


def _eval_metrics(pred):
    raw_predictions, labels = pred
    return 0


def tokenize(tokenizer, prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding=False,
        return_tensors=None,
    )
    if result["input_ids"][-1] != tokenizer.eos_token_id and add_eos_token:
        if len(result["input_ids"]) >= tokenizer.model_max_length:
            result["input_ids"] = result["input_ids"][:-1]
            result["attention_mask"] = result["attention_mask"][:-1]
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def _process_data(data, tokenizer, job_config):
    data = data.to_pandas()
    data = data.fillna("")

    data = data[[TEXT_COLUMN]]
    if job_config.add_eos_token:
        data[TEXT_COLUMN] = data[TEXT_COLUMN] + tokenizer.eos_token
    data = Dataset.from_pandas(data)
    return data


def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


@utils.job_watcher
def train(co2_tracker, payload, huggingface_token, model_path):
    # create model repo
    model_repo = utils.create_repo(
        project_name=payload["proj_name"],
        autotrain_user=payload["username"],
        huggingface_token=huggingface_token,
        model_path=model_path,
    )

    data_path = f"{payload['username']}/autotrain-data-{payload['proj_name']}"
    data = load_dataset(data_path, use_auth_token=huggingface_token)
    logger.info(f"Loaded data from {data_path}")
    job_config = payload["config"]["params"][0]
    job_config["model_name"] = payload["config"]["hub_model"]

    train_data = data["train"]
    valid_data = data["validation"]

    model_name = job_config["model_name"]
    del job_config["model_name"]

    job_config = LMTrainingParams(**job_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=huggingface_token)

    if tokenizer.model_max_length > 2048:
        tokenizer.model_max_length = 2048

    m_arch = utils.get_model_architecture(model_name).lower()
    logger.info(f"Model architecture: {m_arch}")

    use_peft = False
    use_int8 = False

    if "llama" in m_arch or "rwforcausallm" in m_arch:
        use_peft = True
        use_int8 = True

    if "gptneo" in m_arch:
        use_peft = True
        use_int8 = False

    # process data
    train_data = _process_data(data=train_data, tokenizer=tokenizer, job_config=job_config)
    valid_data = _process_data(data=valid_data, tokenizer=tokenizer, job_config=job_config)

    model_config = AutoConfig.from_pretrained(
        model_name,
        use_auth_token=huggingface_token,
        trust_remote_code=True,
    )
    logger.info(model_config)
    if use_peft:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=model_config,
                use_auth_token=huggingface_token,
                torch_dtype=torch.float16,
                load_in_8bit=use_int8,
                device_map="auto",
                trust_remote_code=True,
            )
        except OSError:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=model_config,
                use_auth_token=huggingface_token,
                from_tf=True,
                torch_dtype=torch.float16,
                load_in_8bit=use_int8,
                device_map="auto",
                trust_remote_code=True,
            )
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=model_config,
                use_auth_token=huggingface_token,
                trust_remote_code=True,
            )
        except OSError:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=model_config,
                use_auth_token=huggingface_token,
                from_tf=True,
                trust_remote_code=True,
            )

    # PEFT:
    model.resize_token_embeddings(len(tokenizer))

    if use_peft:
        if use_int8:
            model = prepare_model_for_int8_training(model)
        peft_config = LoraConfig(
            r=job_config.lora_r,
            lora_alpha=job_config.lora_alpha,
            lora_dropout=job_config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ]
            if "rwforcausallm" in m_arch
            else None,
        )
        model = get_peft_model(model, peft_config)

    if job_config.block_size == -1:
        job_config.block_size = None

    if job_config.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if job_config.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({job_config['block_size']}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(job_config.block_size, tokenizer.model_max_length)

    logger.info(model)

    def tokenize_function(examples):
        output = tokenizer(examples[TEXT_COLUMN])
        return output

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    train_data = train_data.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=list(train_data.features),
        desc="Running tokenizer on train dataset",
    )

    valid_data = valid_data.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=list(valid_data.features),
        desc="Running tokenizer on validation dataset",
    )

    train_data = train_data.map(
        group_texts,
        batched=True,
        num_proc=4,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    valid_data = valid_data.map(
        group_texts,
        batched=True,
        num_proc=4,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    logger.info("creating trainer")
    # trainer specific
    logging_steps = int(0.2 * len(valid_data) / job_config.train_batch_size)
    if logging_steps == 0:
        logging_steps = 1

    training_args = dict(
        output_dir=model_path,
        per_device_train_batch_size=job_config.train_batch_size,
        per_device_eval_batch_size=2 * job_config.train_batch_size,
        learning_rate=job_config.learning_rate,
        num_train_epochs=job_config.num_train_epochs,
        evaluation_strategy="epoch",
        logging_steps=logging_steps,
        save_total_limit=1,
        save_strategy="epoch",
        disable_tqdm=not bool(os.environ.get("ENABLE_TQDM", 0)),
        gradient_accumulation_steps=job_config.gradient_accumulation_steps,
        report_to="none",
        auto_find_batch_size=True,
        lr_scheduler_type=job_config.scheduler,
        optim=job_config.optimizer,
        warmup_ratio=job_config.warmup_ratio,
        weight_decay=job_config.weight_decay,
        max_grad_norm=job_config.max_grad_norm,
        fp16=True,
    )

    args = TrainingArguments(**training_args)

    trainer_args = dict(
        args=args,
        model=model,
    )

    data_collator = default_data_collator
    trainer = Trainer(
        **trainer_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    model.config.use_cache = False
    trainer.train()

    logger.info("Finished training")
    logger.info(trainer.state.best_metric)
    eval_scores = trainer.evaluate()

    # create and save model card
    co2_consumed = co2_tracker.stop()
    co2_consumed = co2_consumed * 1000 if co2_consumed is not None else 0

    eval_scores = [f"{k[len('eval_'):]}: {v}" for k, v in eval_scores.items() if k in EVAL_METRICS]
    eval_scores = "\n\n".join(eval_scores)
    model_card = MODEL_CARD.format(
        language=payload["config"]["language"],
        dataset=data_path,
        co2=co2_consumed,
        validation_metrics=eval_scores,
    )
    logger.info(model_card)
    utils.save_model_card(model_card, model_path)

    utils.create_file(
        filename="handler.py",
        file_content=HANDLER_CONTENT_PEFT.strip() if use_peft else HANDLER_CONTENT.strip(),
        model_path=model_path,
    )
    utils.create_file(filename="requirements.txt", file_content=REQUIREMENTS.strip(), model_path=model_path)

    # save model, tokenizer and config
    model = utils.update_model_config(trainer.model, job_config)
    utils.save_tokenizer(tokenizer, model_path)
    utils.save_model(model, model_path)
    utils.remove_checkpoints(model_path=model_path)

    # push model to hub
    logger.info("Pushing model to Hub")
    model_repo.git_pull()
    model_repo.git_add()
    model_repo.git_commit(commit_message="Commit From AutoTrain")
    model_repo.git_push()
