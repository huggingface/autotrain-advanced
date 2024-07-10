import argparse
import json
import torch
from accelerate.state import PartialState
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi
from dataclasses import dataclass, field
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback
import habana_frameworks.torch.hpu as hthpu
#from autotrain import logger
from autotrain.trainers.common import (
    ALLOW_REMOTE_CODE,
    LossLoggingCallback,
    TrainStartCallback,
    UploadLogs,
    monitor,
    pause_space,
    remove_autotrain_data,
    save_training_params,
)
from autotrain.trainers.text_classification import utils
from autotrain.trainers.text_classification.dataset import TextClassificationDataset
from autotrain.trainers.text_classification.params import TextClassificationParams

from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments
from optimum.habana.utils import set_seed
from typing import Optional

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments
from optimum.habana.utils import set_seed

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def parse_args():
    # get training_config.json from the end user
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=False)

    parser.add_argument("--output_dir", type=str, required=False)

    parser.add_argument("--model_name_or_path", type=str, required=False)

    parser.add_argument("--dataset_name", type=str, required=False)

    parser.add_argument("--use_habana", type=str, required=False)

    parser.add_argument("--hub_token", type=str, required=False)


    return parser.parse_args()

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    problem_type: Optional[str] = field(
        default="single_label_classification",
        metadata={"help": "Problem type, such as single_label_classification or multi_label_classification"},
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    add_pad_token: bool = field(
        default=False,
        metadata={"help": "Will add `pad_token` to tokenizer and model's config as `eos_token` if it's not defined."},
    )


# @monitor
def train(config):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GaudiTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(hthpu.is_available())
    device = torch.device("hpu")
    # if isinstance(config, dict):
    #     config = TextClassificationParams(**config)
    train_data = None
    valid_data = None
    # check if config.train_split.csv exists in config.data_path
    # if config.train_split is not None:
    #     if config.data_path == f"{config.project_name}/autotrain-data":
    #         train_data = load_from_disk(config.data_path)[config.train_split]
    #     else:
    #         if ":" in config.train_split:
    #             dataset_config_name, split = config.train_split.split(":")
    #             train_data = load_dataset(
    #                 config.data_path,
    #                 name=dataset_config_name,
    #                 split=split,
    #                 token=config.token,
    #             )
    #         else:
    #             train_data = load_dataset(
    #                 config.data_path,
    #                 split=config.train_split,
    #                 token=config.token,
    #             )
    # if config.valid_split is not None:
    #     if config.data_path == f"{config.project_name}/autotrain-data":
    #         valid_data = load_from_disk(config.data_path)[config.valid_split]
    #     else:
    #         if ":" in config.valid_split:
    #             dataset_config_name, split = config.valid_split.split(":")
    #             valid_data = load_dataset(
    #                 config.data_path,
    #                 name=dataset_config_name,
    #                 split=split,
    #                 token=config.token,
    #             )
    #         else:
    #             valid_data = load_dataset(
    #                 config.data_path,
    #                 split=config.valid_split,
    #                 token=config.token,
    #             )

    if config.dataset_name is not None:
        print(config.dataset_name)

        train_data = load_dataset(
            path = config.dataset_name,
            split= "train" 
        )

        test_data = load_dataset(
            path = config.dataset_name,
            split= "test" 
        )
        print(train_data, test_data)

    print("train data", train_data)
    # classes = train_data.features[config.target_column].names
    # label2id = {c: i for i, c in enumerate(classes)}
    # num_classes = len(classes)
    # if num_classes < 2:
    #     raise ValueError("Invalid number of classes. Must be greater than 1.")

    # if config.valid_split is not None:
    #     num_classes_valid = len(valid_data.unique(config.target_column))
    #     if num_classes_valid != num_classes:
    #         raise ValueError(
    #             f"Number of classes in train and valid are not the same. Training has {num_classes} and valid has {num_classes_valid}"
    #         )
    # model_config = AutoConfig.from_pretrained(config.model_name_or_path, num_labels=num_classes)
    # model_config._num_labels = len(label2id)
    # model_config.label2id = label2id
    # model_config.id2label = {v: k for k, v in label2id.items()}
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name_or_path,
            # config=model_config,
            trust_remote_code=ALLOW_REMOTE_CODE,
            token=config.hub_token,
            ignore_mismatched_sizes=True,
        )
    except OSError:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model,
            config=model_config,
            from_tf=True,
            trust_remote_code=ALLOW_REMOTE_CODE,
            token=config.token,
            ignore_mismatched_sizes=True,
        )
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, token=config.hub_token, trust_remote_code=ALLOW_REMOTE_CODE)
    # train_data = TextClassificationDataset(data=train_data, tokenizer=tokenizer, config=config, device = device)
    # if config.valid_split is not None:
    #     valid_data = TextClassificationDataset(data=valid_data, tokenizer=tokenizer, config=config, device = device)

    # if config.logging_steps == -1:
    #     if config.valid_split is not None:
    #         logging_steps = int(0.2 * len(valid_data) / config.batch_size)
    #     else:
    #         logging_steps = int(0.2 * len(train_data) / config.batch_size)
    #     if logging_steps == 0:
    #         logging_steps = 1
    #     if logging_steps > 25:
    #         logging_steps = 25
    #     config.logging_steps = logging_steps
    # else:
    #     logging_steps = config.logging_steps

    # if config.mixed_precision == "fp16":
    #     training_args["fp16"] = True
    # if config.mixed_precision == "bf16":
    #     training_args["bf16"] = True

    # if config.valid_split is not None:
    #     early_stop = EarlyStoppingCallback(
    #         early_stopping_patience=config.early_stopping_patience,
    #         early_stopping_threshold=config.early_stopping_threshold,
    #     )
    #     callbacks_to_use = [early_stop]
    # else:
    #     callbacks_to_use = []

    # callbacks_to_use.extend([UploadLogs(config=config), LossLoggingCallback(), TrainStartCallback()])
    # print("training_args", training_args)
    
    # args = TrainingArguments(**training_args)
    # trainer_args = dict(
    #     args=args,
    #     model=model,
    #     callbacks=callbacks_to_use,
    #     compute_metrics=(
    #         utils._binary_classification_metrics if num_classes == 2 else utils._multi_class_classification_metrics
    #     ),
    # )
   
    gaudi_config = GaudiConfig.from_pretrained(
        "Habana/bert-large-uncased-whole-word-masking",
        cache_dir=None,
        revision="main",
        token=None,
    )


    trainer = GaudiTrainer(
        model=model,
        gaudi_config=gaudi_config,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        #compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        #data_collator=data_collator,
    )
    # trainer = Trainer(
    #     **trainer_args,
    #     train_dataset=train_data,
    #     eval_dataset=valid_data,
    # )
    trainer.remove_callback(PrinterCallback)
    trainer.train()
    print("Finished training, saving model...")
    #logger.info("Finished training, saving model...")
    trainer.save_model(config.project_name)
    tokenizer.save_pretrained(config.project_name)

    model_card = utils.create_model_card(config, trainer, num_classes)

    # save model card to output directory as README.md
    with open(f"{config.project_name}/README.md", "w") as f:
        f.write(model_card)

    if config.push_to_hub:
        if PartialState().process_index == 0:
            remove_autotrain_data(config)
            save_training_params(config)
            #logger.info("Pushing model to hub...")
            api = HfApi(token=config.token)
            api.create_repo(
                repo_id=f"{config.username}/{config.project_name}", repo_type="model", private=True, exist_ok=True
            )
            api.upload_folder(
                folder_path=config.project_name,
                repo_id=f"{config.username}/{config.project_name}",
                repo_type="model",
            )

    if PartialState().process_index == 0:
        pause_space(config)


if __name__ == "__main__":
    args = parse_args()
    print(f"========args: {args}========")
    # training_config = json.load(open(args.training_config))
    # print(f"training_config {training_config}")
    # config = TextClassificationParams(**training_config)
    #config["data_path"] = "stanfordnlp/imdb"
    # print(f"config{config}")
    train(args)
