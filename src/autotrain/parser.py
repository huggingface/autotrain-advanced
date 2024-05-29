import os
from dataclasses import dataclass

import requests
import yaml

from autotrain import logger
from autotrain.cli.utils import (
    dreambooth_munge_data,
    img_clf_munge_data,
    img_obj_detect_munge_data,
    llm_munge_data,
    sent_transformers_munge_data,
    seq2seq_munge_data,
    tabular_munge_data,
    text_clf_munge_data,
    text_reg_munge_data,
    token_clf_munge_data,
)
from autotrain.project import AutoTrainProject
from autotrain.tasks import TASKS
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.object_detection.params import ObjectDetectionParams
from autotrain.trainers.sent_transformers.params import SentenceTransformersParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.text_regression.params import TextRegressionParams
from autotrain.trainers.token_classification.params import TokenClassificationParams


@dataclass
class AutoTrainConfigParser:
    config_path: str

    def __post_init__(self):
        if self.config_path.startswith("http"):
            response = requests.get(self.config_path)
            if response.status_code == 200:
                self.config = yaml.safe_load(response.content)
            else:
                raise ValueError("Failed to retrieve YAML file.")
        else:
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)

        self.task_param_map = {
            "lm_training": LLMTrainingParams,
            "dreambooth": DreamBoothTrainingParams,
            "image_binary_classification": ImageClassificationParams,
            "image_multi_class_classification": ImageClassificationParams,
            "image_object_detection": ObjectDetectionParams,
            "seq2seq": Seq2SeqParams,
            "tabular": TabularParams,
            "text_binary_classification": TextClassificationParams,
            "text_multi_class_classification": TextClassificationParams,
            "text_single_column_regression": TextRegressionParams,
            "text_token_classification": TokenClassificationParams,
            "sentence_transformers": SentenceTransformersParams,
        }
        self.munge_data_map = {
            "lm_training": llm_munge_data,
            "dreambooth": dreambooth_munge_data,
            "tabular": tabular_munge_data,
            "seq2seq": seq2seq_munge_data,
            "image_multi_class_classification": img_clf_munge_data,
            "image_object_detection": img_obj_detect_munge_data,
            "text_multi_class_classification": text_clf_munge_data,
            "text_token_classification": token_clf_munge_data,
            "text_single_column_regression": text_reg_munge_data,
            "sentence_transformers": sent_transformers_munge_data,
        }
        self.task_aliases = {
            "llm": "lm_training",
            "llm-sft": "lm_training",
            "llm-orpo": "lm_training",
            "llm-generic": "lm_training",
            "llm-dpo": "lm_training",
            "llm-reward": "lm_training",
            "dreambooth": "dreambooth",
            "image_binary_classification": "image_multi_class_classification",
            "image-binary-classification": "image_multi_class_classification",
            "image_classification": "image_multi_class_classification",
            "image-classification": "image_multi_class_classification",
            "seq2seq": "seq2seq",
            "tabular": "tabular",
            "text_binary_classification": "text_multi_class_classification",
            "text-binary-classification": "text_multi_class_classification",
            "text_classification": "text_multi_class_classification",
            "text-classification": "text_multi_class_classification",
            "text_single_column_regression": "text_single_column_regression",
            "text-single-column-regression": "text_single_column_regression",
            "text_regression": "text_single_column_regression",
            "text-regression": "text_single_column_regression",
            "token_classification": "text_token_classification",
            "token-classification": "text_token_classification",
            "image_object_detection": "image_object_detection",
            "image-object-detection": "image_object_detection",
            "object_detection": "image_object_detection",
            "object-detection": "image_object_detection",
            "st:pair": "sentence_transformers",
            "st:pair_class": "sentence_transformers",
            "st:pair_score": "sentence_transformers",
            "st:triplet": "sentence_transformers",
            "st:qa": "sentence_transformers",
            "sentence-transformers:pair": "sentence_transformers",
            "sentence-transformers:pair_class": "sentence_transformers",
            "sentence-transformers:pair_score": "sentence_transformers",
            "sentence-transformers:triplet": "sentence_transformers",
            "sentence-transformers:qa": "sentence_transformers",
        }
        task = self.config.get("task")
        self.task = self.task_aliases.get(task, task)
        if self.task is None:
            raise ValueError("Task is required in the configuration file")
        if self.task not in TASKS:
            raise ValueError(f"Task `{self.task}` is not supported")
        self.backend = self.config.get("backend")
        if self.backend is None:
            raise ValueError("Backend is required in the configuration file")

        logger.info(f"Running task: {self.task}")
        logger.info(f"Using backend: {self.backend}")

        self.parsed_config = self._parse_config()

    def _parse_config(self):
        params = {
            "model": self.config["base_model"],
            "project_name": self.config["project_name"],
        }

        if self.task == "dreambooth":
            params["image_path"] = self.config["data"]["path"]
            params["prompt"] = self.config["data"]["prompt"]
        else:
            params["data_path"] = self.config["data"]["path"]

        if self.task == "lm_training":
            params["chat_template"] = self.config["data"]["chat_template"]
            if "-" in self.config["task"]:
                params["trainer"] = self.config["task"].split("-")[1]
                if params["trainer"] == "generic":
                    params["trainer"] = "default"
                if params["trainer"] not in ["sft", "orpo", "dpo", "reward", "default"]:
                    raise ValueError("Invalid LLM training task")

        if self.task == "sentence_transformers":
            params["trainer"] = self.config["task"].split(":")[1]

        if self.task != "dreambooth":
            for k, v in self.config["data"]["column_mapping"].items():
                params[k] = v
            params["train_split"] = self.config["data"]["train_split"]
            params["valid_split"] = self.config["data"]["valid_split"]
            params["log"] = self.config["log"]

        if "hub" in self.config:
            params["username"] = self.config["hub"]["username"]
            params["token"] = self.config["hub"]["token"]
            params["push_to_hub"] = self.config["hub"]["push_to_hub"]
        else:
            params["username"] = None
            params["token"] = None
            params["push_to_hub"] = False

        if params["username"]:
            if params["username"].startswith("${"):
                params["username"] = os.environ.get(params["username"][2:-1])

        if params["token"]:
            if params["token"].startswith("${"):
                params["token"] = os.environ.get(params["token"][2:-1])

        other_params = self.config.get("params")
        if other_params:
            params.update(other_params)

        return params

    def run(self):
        _params = self.task_param_map[self.task](**self.parsed_config)
        logger.info(_params)
        _munge_fn = self.munge_data_map[self.task]
        _munge_fn(_params, local=self.backend.startswith("local"))
        project = AutoTrainProject(params=_params, backend=self.backend)
        job_id = project.create()
        logger.info(f"Job ID: {job_id}")
