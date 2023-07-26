"""
Copyright 2023 The HuggingFace Team
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from codecarbon import EmissionsTracker
from loguru import logger

from autotrain.dataset import AutoTrainDataset, AutoTrainDreamboothDataset, AutoTrainImageClassificationDataset
from autotrain.languages import SUPPORTED_LANGUAGES
from autotrain.tasks import TASKS
from autotrain.utils import http_get, http_post


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
