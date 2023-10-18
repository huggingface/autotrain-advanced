"""
Copyright 2023 The HuggingFace Team
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import pandas as pd
from codecarbon import EmissionsTracker

from autotrain import logger
from autotrain.backend import SpaceRunner
from autotrain.dataset import AutoTrainDataset, AutoTrainDreamboothDataset, AutoTrainImageClassificationDataset
from autotrain.languages import SUPPORTED_LANGUAGES
from autotrain.tasks import TASKS
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.utils import http_get, http_post


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

        if self.task in ("text_multi_class_classification", "text_binary_classification"):
            self.col_map_text = "autotrain_text"
            self.col_map_target = "autotrain_label"
        if self.task == "lm_training":
            self.col_map_text = "autotrain_text"
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
            # "Local": "local",
            # "AutoTrain": "autotrain",
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

        if "trainer" in _params:
            _params["trainer"] = _params["trainer"].lower()

        if "use_fp16" in _params:
            _params["fp16"] = _params["use_fp16"]
            _params.pop("use_fp16")

        if "int4_8" in _params:
            if _params["int4_8"] == "int4":
                _params["use_int4"] = True
                _params["use_int8"] = False
            elif _params["int4_8"] == "int8":
                _params["use_int4"] = False
                _params["use_int8"] = True
            else:
                _params["use_int4"] = False
                _params["use_int8"] = False
            _params.pop("int4_8")

        return _params

    def _munge_params_text_clf(self, job_idx):
        _params = self._munge_common_params(job_idx)
        _params["model"] = self.model_choice
        _params["text_column"] = self.col_map_text
        _params["target_column"] = self.col_map_target
        _params["valid_split"] = "validation"

        if "use_fp16" in _params:
            _params["fp16"] = _params["use_fp16"]
            _params.pop("use_fp16")

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
                _params = LLMTrainingParams.parse_obj(_params)
            elif self.task_id in (1, 2):
                _params = self._munge_params_text_clf(job_idx)
                _params = TextClassificationParams.parse_obj(_params)
            elif self.task_id in (13, 14, 15, 16, 26):
                _params = self._munge_params_tabular(job_idx)
                _params = TabularParams.parse_obj(_params)
            elif self.task_id == 25:
                _params = self._munge_params_dreambooth(job_idx)
                _params = DreamBoothTrainingParams.parse_obj(_params)
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
        if self.backend == "Local":
            raise NotImplementedError
        if self.backend in self.spaces_backends:
            return self.create_spaces()


@dataclass
class Project:
    dataset: Union[AutoTrainDataset, AutoTrainDreamboothDataset, AutoTrainImageClassificationDataset]
    param_choice: Optional[str] = "autotrain"
    hub_model: Optional[str] = None
    job_params: Optional[List[Dict[str, str]]] = None

    def __post_init__(self):
        self.token = self.dataset.token
        self.name = self.dataset.project_name
        self.username = self.dataset.username
        self.task = self.dataset.task

        self.param_choice = self.param_choice.lower()

        if self.hub_model is not None:
            if len(self.hub_model) == 0:
                self.hub_model = None

        if self.job_params is None:
            self.job_params = []

        logger.info(f"🚀🚀🚀 Creating project {self.name}, task: {self.task}")
        logger.info(f"🚀 Using username: {self.username}")
        logger.info(f"🚀 Using param_choice: {self.param_choice}")
        logger.info(f"🚀 Using hub_model: {self.hub_model}")
        logger.info(f"🚀 Using job_params: {self.job_params}")

        if self.token is None:
            raise ValueError("❌ Please login using `huggingface-cli login`")

        if self.hub_model is not None and len(self.job_params) == 0:
            raise ValueError("❌ Job parameters are required when hub model is specified.")

        if self.hub_model is None and len(self.job_params) > 1:
            raise ValueError("❌ Only one job parameter is allowed in AutoTrain mode.")

        if self.param_choice == "autotrain":
            if "source_language" in self.job_params[0] and "target_language" not in self.job_params[0]:
                self.language = self.job_params[0]["source_language"]
                # remove source language from job params
                self.job_params[0].pop("source_language")
            elif "source_language" in self.job_params[0] and "target_language" in self.job_params[0]:
                self.language = f'{self.job_params[0]["target_language"]}2{self.job_params[0]["source_language"]}'
                # remove source and target language from job params
                self.job_params[0].pop("source_language")
                self.job_params[0].pop("target_language")
            else:
                self.language = "unk"

            if "num_models" in self.job_params[0]:
                self.max_models = self.job_params[0]["num_models"]
                self.job_params[0].pop("num_models")
            elif "num_models" not in self.job_params[0] and "source_language" in self.job_params[0]:
                raise ValueError("❌ Please specify num_models in job_params when using AutoTrain model")
        else:
            self.language = "unk"
            self.max_models = len(self.job_params)

    def create_local(self, payload):
        from autotrain.trainers.dreambooth import train_ui as train_dreambooth
        from autotrain.trainers.image_classification import train as train_image_classification
        from autotrain.trainers.lm_trainer import train as train_lm
        from autotrain.trainers.text_classification import train as train_text_classification

        # check if training tracker file exists in /tmp/
        if os.path.exists(os.path.join("/tmp", "training")):
            raise ValueError("❌ Another training job is already running in this workspace.")

        if len(payload["config"]["params"]) > 1:
            raise ValueError("❌ Only one job parameter is allowed in spaces/local mode.")

        model_path = os.path.join("/tmp/model", payload["proj_name"])
        os.makedirs(model_path, exist_ok=True)

        co2_tracker = EmissionsTracker(save_to_file=False)
        co2_tracker.start()
        # create a training tracker file in /tmp/, using touch
        with open(os.path.join("/tmp", "training"), "w") as f:
            f.write("training")

        if payload["task"] in [1, 2]:
            _ = train_text_classification(
                co2_tracker=co2_tracker,
                payload=payload,
                huggingface_token=self.token,
                model_path=model_path,
            )
        elif payload["task"] in [17, 18]:
            _ = train_image_classification(
                co2_tracker=co2_tracker,
                payload=payload,
                huggingface_token=self.token,
                model_path=model_path,
            )
        elif payload["task"] == 25:
            _ = train_dreambooth(
                co2_tracker=co2_tracker,
                payload=payload,
                huggingface_token=self.token,
                model_path=model_path,
            )
        elif payload["task"] == 9:
            _ = train_lm(
                co2_tracker=co2_tracker,
                payload=payload,
                huggingface_token=self.token,
                model_path=model_path,
            )
        else:
            raise NotImplementedError

        # remove the training tracker file in /tmp/, using rm
        os.remove(os.path.join("/tmp", "training"))

    def create(self, local=False):
        """Create a project and return it"""
        logger.info(f"🚀 Creating project {self.name}, task: {self.task}")
        task_id = TASKS.get(self.task)
        if task_id is None:
            raise ValueError(f"❌ Invalid task selected. Please choose one of {TASKS.keys()}")
        language = str(self.language).strip().lower()
        if task_id is None:
            raise ValueError(f"❌ Invalid task specified. Please choose one of {list(TASKS.keys())}")

        if self.hub_model is not None:
            language = "unk"

        if language not in SUPPORTED_LANGUAGES:
            raise ValueError("❌ Invalid language. Please check supported languages in AutoTrain documentation.")

        payload = {
            "username": self.username,
            "proj_name": self.name,
            "task": task_id,
            "config": {
                "advanced": True,
                "autotrain": True if self.param_choice == "autotrain" else False,
                "language": language,
                "max_models": self.max_models,
                "hub_model": self.hub_model,
                "params": self.job_params,
            },
        }
        logger.info(f"🚀 Creating project with payload: {payload}")

        if local is True:
            return self.create_local(payload=payload)

        logger.info(f"🚀 Creating project with payload: {payload}")
        json_resp = http_post(path="/projects/create", payload=payload, token=self.token).json()
        proj_name = json_resp["proj_name"]
        proj_id = json_resp["id"]
        created = json_resp["created"]

        if created is True:
            return proj_id
        raise ValueError(f"❌ Project with name {proj_name} already exists.")

    def approve(self, project_id):
        # Process data
        _ = http_post(
            path=f"/projects/{project_id}/data/start_processing",
            token=self.token,
        ).json()

        logger.info("⏳ Waiting for data processing to complete ...")
        is_data_processing_success = False
        while is_data_processing_success is not True:
            project_status = http_get(
                path=f"/projects/{project_id}",
                token=self.token,
            ).json()
            # See database.database.enums.ProjectStatus for definitions of `status`
            if project_status["status"] == 3:
                is_data_processing_success = True
                logger.info("✅ Data processing complete!")

            time.sleep(3)

        logger.info(f"🚀 Approving project # {project_id}")
        # Approve training job
        _ = http_post(
            path=f"/projects/{project_id}/start_training",
            token=self.token,
        ).json()
