"""
Copyright 2023 The HuggingFace Team
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests
from loguru import logger

from autotrain.languages import SUPPORTED_LANGUAGES
from autotrain.tasks import TASKS
from autotrain.utils import http_post, user_authentication


@dataclass
class Project:
    token: str
    name: str
    username: str
    task: str
    language: str
    max_models: int
    hub_model: Optional[str] = None
    job_params: Optional[List[Dict]] = None

    def __post_init__(self):
        if self.token is None:
            raise ValueError("❌ Please login using `huggingface-cli login`")

    def create(self):
        """Create a project and return it"""
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
                "language": language,
                "max_models": self.max_models,
                "hub_model": self.hub_model,
                "params": self.job_params,
            },
        }
        json_resp = http_post(path="/projects/create", payload=payload, token=self.token).json()
        proj_name = json_resp["proj_name"]
        created = json_resp["created"]
        if created is True:
            return proj_name
        else:
            raise ValueError(f"❌ Project with name {proj_name} already exists.")

    # def get_project(self, name: str, is_eval: bool = False):
    #     """Retrieves a project"""
    #     self._login_from_conf()
    #     if self.username is None:
    #         raise UnauthenticatedError("❌ Credentials not found ! Please login to AutoNLP first.")
    #     if is_eval:
    #         logger.info(f"☁ Retrieving evaluation project '{name}' from AutoNLP...")
    #         try:
    #             json_resp = http_get(path=f"/evaluate/status/{name}", token=self.token).json()
    #         except requests.exceptions.HTTPError as err:
    #             if err.response.status_code == 404:
    #                 raise ValueError(
    #                     f"❌ Evaluation project '{name}' not found. Please create the project using autonlp evaluate"
    #                 )
    #             raise
    #         return format_eval_status(json_resp)

    #     if self._project is None or self._project.name != name:
    #         logger.info(f"☁ Retrieving project '{name}' from AutoNLP...")
    #         try:
    #             json_resp = http_get(path=f"/projects/{self.username}/{name}", token=self.token).json()
    #         except requests.exceptions.HTTPError as err:
    #             if err.response.status_code == 404:
    #                 raise ValueError(f"❌ Project '{name}' not found. Please create the project using create_project")
    #             else:
    #                 raise
    #         self._project = Project.from_json_resp(json_resp, token=self.token)
    #         self._project.refresh()
    #     else:
    #         self._project.refresh()
    #     logger.info(f"✅ Successfully loaded project: '{name}'!")
    #     return self._project

    # def get_metrics(self, project):
    #     self._login_from_conf()
    #     if self.username is None:
    #         raise UnauthenticatedError("❌ Credentials not found ! Please login to AutoNLP first.")
    #     try:
    #         json_resp = http_get(path=f"/projects/{self.username}/{project}", token=self.token).json()
    #     except requests.exceptions.HTTPError as err:
    #         if err.response.status_code == 404:
    #             raise ValueError(f"❌ Project '{project}' not found!") from err
    #         raise
    #     _metrics = Metrics.from_json_resp(
    #         json_resp=json_resp, token=self.token, project_name=project, username=self.username
    #     )
    #     return _metrics.print()

    # def predict(self, project, model_id, input_text):
    #     self._login_from_conf()
    #     if self.username is None:
    #         raise UnauthenticatedError("❌ Credentials not found ! Please login to AutoNLP first.")
    #     try:
    #         repo_name = f"autonlp-{project}-{model_id}"
    #         api_url = f"https://api-inference.huggingface.co/models/{self.username}/{repo_name}"
    #         payload = {"inputs": input_text}
    #         payload = json.dumps(payload)
    #         headers = {"Authorization": f"Bearer {self.token}"}
    #         response = requests.request("POST", api_url, headers=headers, data=payload)
    #         return json.loads(response.content.decode("utf-8"))
    #     except requests.exceptions.HTTPError as err:
    #         if err.response.status_code == 404:
    #             raise ValueError("❌ Model not found.") from err
    #         raise

    # def list_projects(self, username: Optional[str] = None):
    #     self._login_from_conf()
    #     if self.username is None:
    #         raise UnauthenticatedError("❌ Credentials not found ! Please login to AutoNLP first.")
    #     # default to current user if username is not provided
    #     if username is None:
    #         username = self.username

    #     logger.info(f"📄 Retrieving projects of user {username}...")
    #     json_resp = http_get(path=f"/projects/list?username={username}", token=self.token).json()
    #     return [Project.from_json_resp(elt, token=self.token) for elt in json_resp]

    # def estimate(self, num_train_samples: int, proj_name: str) -> dict:
    #     self._login_from_conf()
    #     if self.username is None:
    #         raise UnauthenticatedError("❌ Credentials not found ! Please login to AutoNLP first.")
    #     project = self.get_project(name=proj_name)
    #     return project.estimate_cost(num_train_samples)
