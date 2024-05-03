import os
from dataclasses import dataclass

import yaml

from autotrain.cli.utils import (
    dreambooth_munge_data,
    img_clf_munge_data,
    llm_munge_data,
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
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.text_regression.params import TextRegressionParams
from autotrain.trainers.token_classification.params import TokenClassificationParams


@dataclass
class ConfigParser:
    config_file: str

    def __post_init__(self):
        with open(self.config_file, "r") as f:
            self.config = yaml.safe_load(f)
        self.parsed_config = self._parse_config()
        self.task_param_map = {
            "lm_training": LLMTrainingParams,
            "dreambooth": DreamBoothTrainingParams,
            "image_binary_classification": ImageClassificationParams,
            "image_multi_class_classification": ImageClassificationParams,
            "seq2seq": Seq2SeqParams,
            "tabular": TabularParams,
            "text_binary_classification": TextClassificationParams,
            "text_multi_class_classification": TextClassificationParams,
            "text_single_column_regression": TextRegressionParams,
            "token_classification": TokenClassificationParams,
        }
        self.munge_data_map = {
            "lm_training": llm_munge_data,
            "dreambooth": dreambooth_munge_data,
            "tabular": tabular_munge_data,
            "seq2seq": seq2seq_munge_data,
            "image_binary_classification": img_clf_munge_data,
            "image_multi_class_classification": img_clf_munge_data,
            "text_binary_classification": text_clf_munge_data,
            "text_multi_class_classification": text_clf_munge_data,
            "token_classification": token_clf_munge_data,
            "text_single_column_regression": text_reg_munge_data,
        }

    def _parse_config(self):
        task = self.config.get("task")
        if task is None:
            raise ValueError("Task is required in the configuration file")
        if task not in TASKS:
            raise ValueError(f"Task `{task}` is not supported")
        params = {
            "model": self.config["base_model"],
            "project_name": self.config["project_name"],
            "log": self.config["log"],
        }

        if task == "dreambooth":
            params["image_path"] = self.config["data"]["path"]
        else:
            params["data_path"] = self.config["data"]["path"]

        if task == "lm_training":
            params["chat_template"] = self.config["data"]["chat_template"]

        if task != "dreambooth":
            for k, v in self.config["data"]["column_mapping"].items():
                params[k] = v
            params["train_split"] = self.config["data"]["train_split"]
            params["valid_split"] = self.config["data"]["valid_split"]

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
        backend = self.config.get("backend")
        task = self.config.get("task")
        if backend is None:
            raise ValueError("Backend is required in the configuration file")
        if task is None:
            raise ValueError("Task is required in the configuration file")

        _params = self.task_param_map[self.config["task"]](**self.parsed_config)
        _munge_fn = self.munge_data_map[self.config["task"]]
        _munge_fn(_params, local=backend.startswith("local"))
        project = AutoTrainProject(params=_params, backend=backend)
        _ = project.create()
