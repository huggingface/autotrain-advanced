import os
from dataclasses import dataclass

import requests
import yaml

from autotrain import logger
from autotrain.project import (
    AutoTrainProject,
    ext_qa_munge_data,
    img_clf_munge_data,
    img_obj_detect_munge_data,
    img_reg_munge_data,
    llm_munge_data,
    sent_transformers_munge_data,
    seq2seq_munge_data,
    tabular_munge_data,
    text_clf_munge_data,
    text_reg_munge_data,
    token_clf_munge_data,
    vlm_munge_data,
)
from autotrain.tasks import TASKS
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.extractive_question_answering.params import ExtractiveQuestionAnsweringParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.image_regression.params import ImageRegressionParams
from autotrain.trainers.object_detection.params import ObjectDetectionParams
from autotrain.trainers.sent_transformers.params import SentenceTransformersParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.text_regression.params import TextRegressionParams
from autotrain.trainers.token_classification.params import TokenClassificationParams
from autotrain.trainers.vlm.params import VLMTrainingParams


@dataclass
class AutoTrainConfigParser:
    """
    AutoTrainConfigParser is a class responsible for parsing and validating the yaml configuration
    required to run various tasks in the AutoTrain framework. It supports loading configurations
    from both local files and remote URLs, and maps task aliases to their respective parameters
    and data munging functions.

    Attributes:
        config_path (str): Path or URL to the configuration file.
        config (dict): Parsed configuration data.
        task_param_map (dict): Mapping of task names to their parameter classes.
        munge_data_map (dict): Mapping of task names to their data munging functions.
        task_aliases (dict): Mapping of task aliases to their canonical task names.
        task (str): The resolved task name from the configuration.
        backend (str): The backend specified in the configuration.
        parsed_config (dict): The parsed configuration parameters.

    Methods:
        __post_init__(): Initializes the parser, loads the configuration, and validates required fields.
        _parse_config(): Parses the configuration and extracts relevant parameters based on the task.
        run(): Executes the task with the parsed configuration.
    """

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
            "image_single_column_regression": ImageRegressionParams,
            "vlm": VLMTrainingParams,
            "text_extractive_question_answering": ExtractiveQuestionAnsweringParams,
        }
        self.munge_data_map = {
            "lm_training": llm_munge_data,
            "tabular": tabular_munge_data,
            "seq2seq": seq2seq_munge_data,
            "image_multi_class_classification": img_clf_munge_data,
            "image_object_detection": img_obj_detect_munge_data,
            "text_multi_class_classification": text_clf_munge_data,
            "text_token_classification": token_clf_munge_data,
            "text_single_column_regression": text_reg_munge_data,
            "sentence_transformers": sent_transformers_munge_data,
            "image_single_column_regression": img_reg_munge_data,
            "vlm": vlm_munge_data,
            "text_extractive_question_answering": ext_qa_munge_data,
        }
        self.task_aliases = {
            "llm": "lm_training",
            "llm-sft": "lm_training",
            "llm-orpo": "lm_training",
            "llm-generic": "lm_training",
            "llm-dpo": "lm_training",
            "llm-reward": "lm_training",
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
            "st": "sentence_transformers",
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
            "image_single_column_regression": "image_single_column_regression",
            "image-single-column-regression": "image_single_column_regression",
            "image_regression": "image_single_column_regression",
            "image-regression": "image_single_column_regression",
            "image-scoring": "image_single_column_regression",
            "vlm:captioning": "vlm",
            "vlm:vqa": "vlm",
            "extractive_question_answering": "text_extractive_question_answering",
            "ext_qa": "text_extractive_question_answering",
            "ext-qa": "text_extractive_question_answering",
            "extractive-qa": "text_extractive_question_answering",
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

        if self.task == "vlm":
            params["trainer"] = self.config["task"].split(":")[1]

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
