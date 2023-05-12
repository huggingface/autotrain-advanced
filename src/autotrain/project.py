"""
Copyright 2023 The HuggingFace Team
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from loguru import logger

from autotrain.dataset import AutoTrainDataset, AutoTrainDreamboothDataset, AutoTrainImageClassificationDataset
from autotrain.languages import SUPPORTED_LANGUAGES
from autotrain.tasks import TASKS
from autotrain.utils import http_get, http_post


@dataclass
class Project:
    dataset: Union[AutoTrainDataset, AutoTrainImageClassificationDataset, AutoTrainDreamboothDataset]
    param_choice: Optional[str] = "autotrain"
    hub_model: Optional[str] = None
    job_params: Optional[List[Dict]] = None

    def __post_init__(self):
        self.token = self.dataset.token
        self.name = self.dataset.project_name
        self.username = self.dataset.username
        self.task = self.dataset.task

        self.param_choice = self.param_choice.lower()

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

    def create(self):
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
