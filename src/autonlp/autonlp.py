"""
Copyright 2020 The HuggingFace Team
"""

import json
import os

import requests
from loguru import logger

from . import config
from .languages import SUPPORTED_LANGUAGES
from .metrics import Metrics
from .model import Model
from .project import Project
from .tasks import TASKS
from .utils import UnauthenticatedError, http_get, http_post


class AutoNLP:
    def __init__(self, config_dir: str = None) -> None:
        self.username = None
        self.token = None
        self._project = None
        self.config_dir = config_dir
        if self.config_dir is None:
            home_dir = os.path.expanduser("~")
            self.config_dir = os.path.join(home_dir, ".autonlp")
        os.makedirs(self.config_dir, exist_ok=True)

    def get_token(self):
        """Retrieve API token, or raise UnauthenticatedError"""
        self._login_from_conf()
        if self.token is None:
            raise UnauthenticatedError("❌ Credentials not found ! Please login to AutoNLP first.")
        return self.token

    def login(self, token: str):
        """Login to AutoNLP"""
        if token.startswith("api_org"):
            logger.error("⚠ Authenticating as an organization is not allowed. Please provide a user API key.")
            raise ValueError("Login with an organization API keys are not supported")
        try:
            auth_resp = http_get(path="/whoami-v2", domain=config.HF_API, token=token, token_prefix="Bearer")
        except requests.HTTPError as err:
            if err.response.status_code == 401:
                logger.error("❌ Failed to authenticate. Check the passed token is valid!")
            raise
        user_identity = auth_resp.json()
        self.username = user_identity["name"]
        logger.info(f"🗝 Successfully logged in as {self.username}")
        orgs = []
        if user_identity["type"] == "user":
            orgs = [org["name"] for org in user_identity["orgs"]]
        self.orgs = orgs
        self.token = token
        login_dict = {"username": self.username, "orgs": self.orgs, "token": token}
        logger.info(f"🗝 Storing credentials in:  {self.config_dir}")
        with open(os.path.join(self.config_dir, "autonlp.json"), "w") as fp:
            json.dump(login_dict, fp)

    def _login_from_conf(self):
        """Retrieve credentials from local config"""
        conf_json = None
        if self.username is None or self.token is None:
            logger.info("🗝 Retrieving credentials from config...")
            if os.path.isfile(os.path.join(self.config_dir, "autonlp.json")):
                with open(os.path.join(self.config_dir, "autonlp.json"), "r") as conf_file:
                    conf_json = json.load(conf_file)
                    if conf_json is None:
                        raise UnauthenticatedError("❌ Credentials not found! Please login to AutoNLP first.")
                    else:
                        self.username = conf_json["username"]
                        self.orgs = conf_json["orgs"]
                        self.token = conf_json["token"]

    def create_project(self, name: str, task: str, language: str):
        """Create a project and return it"""
        self._login_from_conf()
        task_id = TASKS.get(task)
        language = str(language).strip().lower()
        if len(language) != 2 or language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"❌ Invalid language selected. Please choose one of {SUPPORTED_LANGUAGES}")
        if task_id is None:
            raise ValueError(f"❌ Invalid task specified. Please choose one of {list(TASKS.keys())}")
        payload = {
            "username": self.username,
            "proj_name": name,
            "task": task_id,
            "config": {"version": 0, "patch": 1, "language": language},
        }
        json_resp = http_post(path="/projects/create", payload=payload, token=self.token).json()
        proj_name = json_resp["proj_name"]
        if json_resp["created"] is True:
            logger.info(f"✅ Successfully created project: '{proj_name}'!")
        else:
            logger.info(f"🤙 Project '{proj_name}' already exists, it was loaded successfully.")
        self._project = Project.from_json_resp(json_resp, token=self.token)
        self._project.refresh()
        return self._project

    def get_project(self, name):
        """Retrieves a project"""
        self._login_from_conf()
        if self.username is None:
            raise UnauthenticatedError("❌ Credentials not found ! Please login to AutoNLP first.")
        if self._project is None or self._project.name != name:
            logger.info(f"☁ Retrieving project '{name}' from AutoNLP...")
            try:
                json_resp = http_get(path=f"/projects/{self.username}/{name}", token=self.token).json()
            except requests.exceptions.HTTPError as err:
                if err.response.status_code == 404:
                    raise ValueError(f"❌ Project '{name}' not found. Please create the project using create_project")
                else:
                    raise
            self._project = Project.from_json_resp(json_resp, token=self.token)
            self._project.refresh()
        else:
            self._project.refresh()
        logger.info(f"✅ Successfully loaded project: '{name}'!")
        return self._project

    def get_metrics(self, model_id, project):
        self._login_from_conf()
        if self.username is None:
            raise UnauthenticatedError("❌ Credentials not found ! Please login to AutoNLP first.")
        if model_id is not None:
            try:
                json_resp = http_get(f"/models/{self.username}/{model_id}", token=self.token).json()
            except requests.exceptions.HTTPError as err:
                if err.response.status_code == 404:
                    raise ValueError(f"❌ Model '{model_id}' not found.") from err
                raise
            _model_info = Model.from_json_resp(
                json_resp=json_resp, token=self.token, username=self.username, model_id=model_id
            )
            return _model_info.print()
        if project is not None:
            try:
                json_resp = http_get(path=f"/projects/{self.username}/{project}", token=self.token).json()
            except requests.exceptions.HTTPError as err:
                if err.response.status_code == 404:
                    raise ValueError(f"❌ Project '{project}' not found!") from err
                raise
            _metrics = Metrics.from_json_resp(
                json_resp=json_resp, token=self.token, project_name=project, username=self.username
            )
            return _metrics.print()

    def predict(self, model_id, input_text):
        self._login_from_conf()
        if self.username is None:
            raise UnauthenticatedError("❌ Credentials not found ! Please login to AutoNLP first.")
        try:
            payload = {"username": self.username, "model_id": model_id, "input_text": input_text}
            json_resp = http_post(path="/models/predict", payload=payload, token=self.token).json()
            return json_resp
        except requests.exceptions.HTTPError as err:
            if err.response.status_code == 404:
                raise ValueError(f"❌ Model '{model_id}' not found.") from err
            raise


if __name__ == "__main__":
    client = AutoNLP()
    client.login(token="TEST_KEY")
    project = client.create_project(name="imdb_test_4", task="binary_classification")
    token = client.get_token()
    # project = client.get_project(name="imdb_test_4")
    col_mapping = {"sentiment": "target", "review": "text"}

    train_files = ["/home/abhishek/datasets/imdb_folds.csv"]
    valid_files = ["/home/abhishek/datasets/imdb_valid.csv"]
    project.upload(train_files, split="train", col_mapping=col_mapping)
    project.upload(valid_files, split="valid", col_mapping=col_mapping)
    project.train()
    project.refresh()
    print(project)
