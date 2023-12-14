"""
Copyright 2023 The HuggingFace Team
"""

import json
from dataclasses import dataclass
from typing import Union

import pandas as pd

from autotrain import logger
from autotrain.backend import SpaceRunner
from autotrain.dataset import AutoTrainDataset, AutoTrainDreamboothDataset, AutoTrainImageClassificationDataset
from autotrain.tasks import TASKS
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams


@dataclass
class AutoTrainProject:
    dataset: Union[AutoTrainDataset, AutoTrainDreamboothDataset, AutoTrainImageClassificationDataset]
    job_params: pd.DataFrame

    def __post_init__(self):
        self.token = self.dataset.token
        self.project_name = self.dataset.project_name
        self.username = self.dataset.username
        self.task = self.dataset.task
        if isinstance(self.dataset, AutoTrainDataset):
            self.col_mapping = self.dataset.column_mapping
        self.data_path = f"{self.username}/autotrain-data-{self.project_name}"

        self.backend = self.job_params.loc[0, "backend"]
        if "model_choice" in self.job_params.columns:
            self.model_choice = self.job_params.loc[0, "model_choice"]
        if "param_choice" in self.job_params.columns:
            self.param_choice = self.job_params.loc[0, "param_choice"]

        self.task_id = TASKS.get(self.task)
        self.num_jobs = len(self.job_params)

        if self.task in ("text_multi_class_classification", "text_binary_classification", "seq2seq"):
            self.col_map_text = "autotrain_text"
            self.col_map_target = "autotrain_label"
        if self.task == "lm_training":
            self.col_map_text = "autotrain_text"
        if self.task == "image_multi_class_classification":
            self.col_map_image = "autotrain_image"
            self.col_map_target = "autotrain_label"
        if self.task.startswith("tabular_"):
            self.col_map_id = "autotrain_id"
            _tabular_target_cols = ["autotrain_label"]
            if isinstance(self.col_mapping["label"], str) or len(self.col_mapping["label"]) > 1:
                _tabular_target_cols = [f"autotrain_label_{i}" for i in range(len(self.col_mapping["label"]))]
            self.col_map_target = _tabular_target_cols

        self.spaces_backends = {
            "A10G Large": "spaces-a10gl",
            "A10G Small": "spaces-a10gs",
            "A100 Large": "spaces-a100",
            "T4 Medium": "spaces-t4m",
            "T4 Small": "spaces-t4s",
            "CPU Upgrade": "spaces-cpu",
            "CPU (Free)": "spaces-cpuf",
            "DGX 1xA100": "dgx-a100",
            "DGX 2xA100": "dgx-2a100",
            "DGX 4xA100": "dgx-4a100",
            "DGX 8xA100": "dgx-8a100",
            "Local": "local",
            "EP US-East-1 1xA10g": "ep-aws-useast1-m",
            "EP US-East-1 1xA100": "ep-aws-useast1-xl",
            "EP US-East-1 2xA100": "ep-aws-useast1-2xl",
            "EP US-East-1 4xA100": "ep-aws-useast1-4xl",
            "EP US-East-1 8xA100": "ep-aws-useast1-8xl",
        }

        self.job_params_json = self.job_params.to_json(orient="records")
        logger.info(self.job_params_json)

    def _munge_common_params(self, job_idx):
        _params = json.loads(self.job_params_json)[job_idx]
        _params["token"] = self.token
        _params["project_name"] = f"{self.project_name}-{job_idx}"
        _params["push_to_hub"] = True
        _params["repo_id"] = f"{self.username}/{self.project_name}-{job_idx}"
        _params["data_path"] = self.data_path
        _params["username"] = self.username
        return _params

    def _munge_params_llm(self, job_idx):
        _params = self._munge_common_params(job_idx)
        _params["model"] = self.model_choice
        _params["text_column"] = self.col_map_text
        _params["prompt_text_column"] = "autotrain_prompt"
        _params["rejected_text_column"] = "autotrain_rejected_text"

        if "trainer" in _params:
            _params["trainer"] = _params["trainer"].lower()

        return _params

    def _munge_params_text_clf(self, job_idx):
        _params = self._munge_common_params(job_idx)
        _params["model"] = self.model_choice
        _params["text_column"] = self.col_map_text
        _params["target_column"] = self.col_map_target
        _params["valid_split"] = "validation"

        return _params

    def _munge_params_seq2seq(self, job_idx):
        _params = self._munge_common_params(job_idx)
        _params["model"] = self.model_choice
        _params["text_column"] = self.col_map_text
        _params["target_column"] = self.col_map_target
        _params["valid_split"] = "validation"

        return _params

    def _munge_params_img_clf(self, job_idx):
        _params = self._munge_common_params(job_idx)
        _params["model"] = self.model_choice
        _params["image_column"] = self.col_map_image
        _params["target_column"] = self.col_map_target
        _params["valid_split"] = "validation"

        return _params

    def _munge_params_tabular(self, job_idx):
        _params = self._munge_common_params(job_idx)
        _params["id_column"] = self.col_map_id
        _params["target_columns"] = self.col_map_target
        _params["valid_split"] = "validation"

        if len(_params["categorical_imputer"].strip()) == 0 or _params["categorical_imputer"].lower() == "none":
            _params["categorical_imputer"] = None
        if len(_params["numerical_imputer"].strip()) == 0 or _params["numerical_imputer"].lower() == "none":
            _params["numerical_imputer"] = None
        if len(_params["numeric_scaler"].strip()) == 0 or _params["numeric_scaler"].lower() == "none":
            _params["numeric_scaler"] = None

        return _params

    def _munge_params_dreambooth(self, job_idx):
        _params = self._munge_common_params(job_idx)
        _params["model"] = self.model_choice
        _params["image_path"] = self.data_path

        if "weight_decay" in _params:
            _params["adam_weight_decay"] = _params["weight_decay"]
            _params.pop("weight_decay")

        return _params

    def create_spaces(self):
        _created_spaces = []
        for job_idx in range(self.num_jobs):
            if self.task_id == 9:
                _params = self._munge_params_llm(job_idx)
                _params = LLMTrainingParams(**_params)
            elif self.task_id in (1, 2):
                _params = self._munge_params_text_clf(job_idx)
                _params = TextClassificationParams(**_params)
            elif self.task_id in (13, 14, 15, 16, 26):
                _params = self._munge_params_tabular(job_idx)
                _params = TabularParams(**_params)
            elif self.task_id == 25:
                _params = self._munge_params_dreambooth(job_idx)
                _params = DreamBoothTrainingParams(**_params)
            elif self.task_id == 28:
                _params = self._munge_params_seq2seq(job_idx)
                _params = Seq2SeqParams(**_params)
            elif self.task_id == 18:
                _params = self._munge_params_img_clf(job_idx)
                _params = ImageClassificationParams(**_params)
            else:
                raise NotImplementedError
            logger.info(f"Creating Space for job: {job_idx}")
            logger.info(f"Using params: {_params}")
            sr = SpaceRunner(params=_params, backend=self.spaces_backends[self.backend])
            space_id = sr.prepare()
            logger.info(f"Space created with id: {space_id}")
            _created_spaces.append(space_id)
        return _created_spaces

    def create(self):
        if self.backend == "AutoTrain":
            raise NotImplementedError
        if self.backend in self.spaces_backends:
            return self.create_spaces()
