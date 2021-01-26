# flake8: noqa
# coding=utf-8
# Copyright 2020 The HuggingFace Team
# Lint as: python3
# pylint: enable=line-too-long

import json
import os
from typing import List, Union

import requests
from loguru import logger

from .project import Project
from .tasks import TASKS
from .utils import UnauthenticatedError, UnreachableAPIError, http_get, http_post


class AutoNLP:
    def __init__(self, config_dir: str = None) -> None:
        self.username = None
        self.api_key = None
        self._project = None
        self.config_dir = config_dir
        if self.config_dir is None:
            home_dir = os.path.expanduser("~")
            self.config_dir = os.path.join(home_dir, ".autonlp")
        os.makedirs(self.config_dir, exist_ok=True)

    def get_token(self):
        """Retrieve API token, or raise UnauthenticatedError"""
        self._login_from_conf()
        if self.api_key is None:
            raise UnauthenticatedError("❌ Credentials not found ! Please login to AutoNLP first.")
        return self.api_key

    def login(self, username: str, api_key: str):
        """Login to AutoNLP"""
        self.username = username
        self.api_key = api_key
        # verify the user here and get the api key
        # save the api key in a json file
        login_dict = {"username": self.username, "api_key": api_key}
        # TODO: these credentials need to be passed with every request to the backend
        logger.info(f"🗝 Storing credentials in:  {self.config_dir}")
        with open(os.path.join(self.config_dir, "autonlp.json"), "w") as fp:
            json.dump(login_dict, fp)

    def _login_from_conf(self):
        """Retrieve credentials from local config"""
        conf_json = None
        if self.username is None or self.api_key is None:
            logger.info("🗝 Retrieving credentials from config...")
            if os.path.isfile(os.path.join(self.config_dir, "autonlp.json")):
                with open(os.path.join(self.config_dir, "autonlp.json"), "r") as conf_file:
                    conf_json = json.load(conf_file)
                    if conf_json is None:
                        raise UnauthenticatedError("❌ Credentials not found ! Please login to AutoNLP first.")
                    else:
                        self.username = conf_json["username"]
                        self.api_key = conf_json["api_key"]

    def create_project(self, name: str, task: str):
        """Create a project and return it"""
        self._login_from_conf()
        task_id = TASKS.get(task)
        if task_id is None:
            raise ValueError(f"❌ Invalid task specified. Please choose one of {list(TASKS.keys())}")
        payload = {
            "username": self.username,
            "proj_name": name,
            "task": task_id,
            "config": {"version": 0, "patch": 1},
        }
        json_resp = http_post(path="/projects", payload=payload, token=self.api_key).json()
        proj_name = json_resp["proj_name"]
        if json_resp["created"] is True:
            logger.info(f"✅ Successfully created project: '{proj_name}'!")
        else:
            logger.info(f"🤙 Project '{proj_name}' already exists, it was loaded successfully.")
        self._project = Project.from_json_resp(json_resp)
        return self._project

    def get_project(self, name):
        """Retrieves a project"""
        self._login_from_conf()
        if self.username is None:
            raise UnauthenticatedError("❌ Credentials not found ! Please login to AutoNLP first.")
        if self._project is None or self._project.name != name:
            logger.info(f"☁ Retrieving project '{name}' from AutoNLP...")
            json_resp = http_get(path=f"/projects/{self.username}/{name}", token=self.api_key).json()
            proj_id = json_resp.get("id")
            if proj_id is None:
                raise ValueError(
                    f"❌ Project '{self._project.name}' not found. Please create the project using create_project"
                )
            else:
                self._project = Project.from_json_resp(json_resp)
        logger.info(f"✅ Successfully loaded project: '{name}'!")
        return self._project


if __name__ == "__main__":
    client = AutoNLP()
    client.login(username="abhishek", api_key="TEST_KEY")
    project = client.create_project(name="imdb_test_4", task="binary_classification")
    token = client.get_token()
    # project = client.get_project(name="imdb_test_4")
    col_mapping = {"sentiment": "target", "review": "text"}

    train_files = ["/home/abhishek/datasets/imdb_folds.csv"]
    valid_files = ["/home/abhishek/datasets/imdb_valid.csv"]
    project.upload(train_files, split="train", col_mapping=col_mapping, token=token)
    project.upload(valid_files, split="valid", col_mapping=col_mapping, token=token)

    project.train()
