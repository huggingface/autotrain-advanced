"""
Copyright 2020 The HuggingFace Team
"""

import json
import os
from typing import List, Optional

import requests
from loguru import logger

from .auth import ForbiddenError, login, login_from_conf, select_identity
from .evaluate import Evaluate, format_datasets_task, format_eval_status, get_dataset_splits
from .languages import SUPPORTED_LANGUAGES
from .metrics import Metrics
from .project import Project
from .tasks import DATASETS_TASKS, TASKS
from .utils import http_get, http_post


class AutoNLP:
    def __init__(self, config_dir: str = None) -> None:
        self._project = None
        self._eval_proj = None
        self.config_dir = config_dir
        if self.config_dir is None:
            home_dir = os.path.expanduser("~")
            self.config_dir = os.path.join(home_dir, ".autonlp")
        os.makedirs(self.config_dir, exist_ok=True)

    def get_token(self) -> str:
        """Retrieve API token

        Raises:
            :class:``.auth.NotAuthenticatedError``:
                Not authenticated (you need to login first)
            :class:``.auth.AuthenticationError``:
                Failed to authenticate
        """
        login_info = login_from_conf(save_dir=self.config_dir)
        return login_info["token"]

    def login(self, token: str):
        """Login to AutoNLP

        Raises:
            :class:``.auth.AuthenticationError``:
                Failed to authenticate
            :class:``requests.HTTPError``:
                Failed to reach AutoNLP's API
        """
        login(token, save_dir=self.config_dir)

    def switch_identity(self, new_identity: str):
        select_identity(new_identity=new_identity, save_dir=self.config_dir)

    def create_project(
        self,
        name: str,
        task: str,
        language: str,
        max_models: int,
        hub_model: Optional[str] = None,
        username: Optional[str] = None,
    ) -> Project:
        """Create a project and return it"""
        login_info = login_from_conf(save_dir=self.config_dir)
        if username is None:
            username = login_info["selected_identity"]
            logger.warning(f"Creating project under identity: '{username}'")
        if username not in [identity.name for identity in login_info["identities"]]:
            raise ForbiddenError(f"Cannot impersonate '{username}'")

        task_id = TASKS.get(task)
        if task_id is None:
            raise ValueError(f"❌ Invalid task selected. Please choose one of {TASKS.keys()}")

        language = str(language).strip().lower()
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError("❌ Invalid language selected. Please check supported languages in AutoNLP documentation.")

        payload = {
            "username": username,
            "proj_name": name,
            "task": task_id,
            "config": {
                "language": language,
                "max_models": max_models,
                "hub_model": hub_model,
            },
        }
        json_resp = http_post(path="/projects/create", payload=payload, token=login_info["token"]).json()
        proj_username, proj_name = json_resp["username"], json_resp["proj_name"]
        created = json_resp["created"]
        if created is True:
            logger.info(f"✅ Successfully created project: '{proj_username}/{proj_name}'!")
        else:
            logger.info(f"🤙 Project '{proj_username}{proj_name}' already exists, it was loaded successfully.")
        self._project = Project.from_json_resp(json_resp, token=login_info["token"])
        self._project.refresh()
        return self._project

    def create_evaluation(
        self,
        task: str,
        dataset: str,
        model: str,
        split: str,
        col_mapping: Optional[str] = None,
        config: Optional[str] = None,
        username: Optional[str] = None,
    ) -> Evaluate:
        login_info = login_from_conf(save_dir=self.config_dir)
        if username is None:
            username = login_info["selected_identity"]
            logger.warning(f"Creating evaluation project under identity: '{username}'")
        if username not in [identity.name for identity in login_info["identities"]]:
            raise ForbiddenError(f"Cannot impersonate '{username}'")

        splits = get_dataset_splits(dataset=dataset, config=config)
        if split not in splits:
            raise ValueError(f"❌ Split {split} not found in dataset {dataset}. Available splits: {splits}")

        if task in DATASETS_TASKS:
            task = format_datasets_task(task, dataset, config)
            if col_mapping:
                logger.warning("A task template from `datasets` has been selected. Deleting `col_mapping` ...")
                col_mapping = None
        elif col_mapping is None:
            raise ValueError(
                f"❌ A column mapping must be provided for task {task}. Please provide a value for `col_mapping`."
            )

        task_id = TASKS.get(task)
        if task_id is None:
            raise ValueError(f"❌ Invalid task selected. Please choose one of {TASKS.keys()}")

        mapping_dict = {}
        if col_mapping:
            for c_m in col_mapping.strip().split(","):
                k, v = c_m.split(":")
                mapping_dict[k] = v

        payload = {
            "username": login_info["selected_identity"],
            "dataset": dataset,
            "task": task_id,
            "model": model,
            "col_mapping": mapping_dict,
            "split": split,
            "config": config,
        }
        json_resp = http_post(path="/evaluate/create", payload=payload, token=login_info["token"]).json()
        self._eval_proj = Evaluate.from_json_resp(json_resp, token=login_info["token"])
        return self._eval_proj

    def create_benchmark(
        self,
        dataset: str,
        submission: str,
        eval_name: str,
        username: Optional[str] = None,
    ):
        login_info = login_from_conf(save_dir=self.config_dir)
        if username is None:
            username = login_info["selected_identity"]
            logger.warning(f"Creating benhchmark project under identity: '{username}'")
        if username not in [identity.name for identity in login_info["identities"]]:
            raise ForbiddenError(f"Cannot impersonate '{username}'")

        task_id = 1
        payload = {
            "username": username,
            "dataset": dataset,
            "task": task_id,
            "model": eval_name,
            "submission_dataset": submission,
            "col_mapping": {},
            "split": "test",
            "config": None,
        }
        json_resp = http_post(path="/evaluate/create", payload=payload, token=login_info["token"]).json()
        self._eval_proj = Evaluate.from_json_resp(json_resp, token=login_info["token"])
        return self._eval_proj

    def get_eval_job_status(self, eval_job_id: int) -> int:
        json_resp = http_get(path=f"/evaluate/status/{eval_job_id}", token=self.get_token()).json()
        return json_resp["status"]

    def get_project(self, name: str, username: Optional[str] = None, is_eval: bool = False):
        """Retrieves a project"""
        login_info = login_from_conf(save_dir=self.config_dir)
        if username is None:
            username = login_info["selected_identity"]
            logger.warning(f"Creating benhchmark project under identity: '{username}'")
        if username not in [identity.name for identity in login_info["identities"]]:
            raise ForbiddenError(f"Cannot impersonate '{username}'")

        if is_eval:
            logger.info(f"☁ Retrieving evaluation project '{name}' from AutoNLP...")
            try:
                json_resp = http_get(path=f"/evaluate/status/{name}", token=login_info["token"]).json()
            except requests.exceptions.HTTPError as err:
                if err.response.status_code == 404:
                    raise ValueError(
                        f"❌ Evaluation project '{name}' not found. Please create the project using autonlp evaluate"
                    )
                raise
            return format_eval_status(json_resp)

        if self._project is None or self._project.name != name or self._project.user != username:
            logger.info(f"☁ Retrieving project '{name}' from AutoNLP...")
            try:
                json_resp = http_get(path=f"/projects/{username}/{name}", token=login_info["token"]).json()
            except requests.exceptions.HTTPError as err:
                if err.response.status_code == 404:
                    raise ValueError(f"❌ Project '{name}' not found. Please create the project using create_project")
                else:
                    raise
            self._project = Project.from_json_resp(json_resp, token=login_info["token"])
            self._project.refresh()
        else:
            self._project.refresh()
        logger.info(f"✅ Successfully loaded project: '{name}'!")
        return self._project

    def get_metrics(self, project: str, username: Optional[str] = None):
        login_info = login_from_conf(self.config_dir)
        if username is None:
            username = login_info["selected_identity"]
            logger.warning(f"Creating benhchmark project under identity: '{username}'")
        if username not in [identity.name for identity in login_info["identities"]]:
            raise ForbiddenError(f"Cannot impersonate '{username}'")

        try:
            json_resp = http_get(path=f"/projects/{username}/{project}", token=login_info["token"]).json()
        except requests.exceptions.HTTPError as err:
            if err.response.status_code == 404:
                raise ValueError(f"❌ Project '{project}' not found!") from err
            raise
        _metrics = Metrics.from_json_resp(
            json_resp=json_resp, token=login_info["token"], project_name=project, username=username
        )
        return _metrics.print()

    def predict(self, project: str, model_id: int, input_text: str, username: Optional[str] = None):
        login_info = login_from_conf(self.config_dir)
        if username is None:
            username = login_info["selected_identity"]
            logger.warning(f"Creating benhchmark project under identity: '{username}'")
        if username not in [identity.name for identity in login_info["identities"]]:
            raise ForbiddenError(f"Cannot impersonate '{username}'")

        try:
            repo_name = f"autonlp-{project}-{model_id}"
            api_url = f"https://api-inference.huggingface.co/models/{username}/{repo_name}"
            payload = {"inputs": input_text}
            payload = json.dumps(payload)
            headers = {"Authorization": f"Bearer {login_info['token']}"}
            response = requests.request("POST", api_url, headers=headers, data=payload)
            return json.loads(response.content.decode("utf-8"))
        except requests.exceptions.HTTPError as err:
            if err.response.status_code == 404:
                raise ValueError("❌ Model not found.") from err
            raise

    def list_projects(self, username: Optional[str] = None) -> List[Project]:
        login_info = login_from_conf()
        if username is None:
            username = login_info["selected_identity"]
        if username not in [identity.name for identity in login_info["identities"]]:
            raise ForbiddenError(f"Cannot impersonate '{username}'")

        logger.info(f"📄 Retrieving projects of user {username}...")
        json_resp = http_get(path=f"/projects/list?username={username}", token=login_info["token"]).json()
        return [Project.from_json_resp(elt, token=login_info["token"]) for elt in json_resp]

    def estimate(self, num_train_samples: int, proj_name: str, username: Optional[str] = None) -> dict:
        login_info = login_from_conf()
        if username is None:
            username = login_info["selected_identity"]
        if username not in [identity.name for identity in login_info["identities"]]:
            raise ForbiddenError(f"Cannot impersonate '{username}'")
        project = self.get_project(name=proj_name)
        return project.estimate_cost(num_train_samples)
