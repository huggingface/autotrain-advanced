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

from . import config
from .project import Project
from .tasks import TASKS
from .config import get_auth_headers


class AutoNLP:
    def __init__(self, config_dir: str = None) -> None:
        self.username = None
        self.api_key = None
        self.project = None
        self.config_dir = config_dir
        if self.config_dir is None:
            home_dir = os.path.expanduser("~")
            self.config_dir = os.path.join(home_dir, ".autonlp")
        os.makedirs(self.config_dir, exist_ok=True)

    def login(self, username, api_key):
        self.username = username
        self.api_key = api_key
        # verify the user here and get the api key
        # save the api key in a json file
        login_dict = {"username": self.username, "api_key": api_key}
        # TODO: these credentials need to be passed with every request to the backend
        logger.info(f"Storing credentials in:  {self.config_dir}")
        with open(os.path.join(self.config_dir, "autonlp.json"), "w") as fp:
            json.dump(login_dict, fp)

    def _login_from_conf(self):
        conf_json = None
        if self.username is None or self.api_key is None:
            if os.path.isfile(os.path.join(self.config_dir, "autonlp.json")):
                with open(os.path.join(self.config_dir, "autonlp.json"), "r") as conf_file:
                    conf_json = json.load(conf_file)
                    if conf_json is None:
                        raise Exception("Unable to login / credentials not found. Please login first")
                    else:
                        self.username = conf_json["username"]
                        self.api_key = conf_json["api_key"]

    def create_project(self, name: str, task: str):
        self._login_from_conf()
        task_id = TASKS.get(task, -1)
        if task_id == -1:
            raise Exception(f"Invalid task specified. Please choose one of {list(TASKS.keys())}")
        payload = {
            "username": self.username,
            "proj_name": name,
            "task": task_id,
            "config": {"version": 0, "patch": 1},
        }
        try:
            resp = requests.post(
                url=config.HF_AUTONLP_BACKEND_API + "/projects/",
                json=payload,
                headers=get_auth_headers(token=self.api_key),
            )
        except requests.exceptions.ConnectionError:
            raise Exception("API is currently not available")
        resp_json = resp.json()
        logger.info(resp_json)

        if resp_json["created"] is True:
            logger.info(f"Created project: {resp_json['proj_name']}")
        else:
            logger.info(f"Project already exists. Loaded successfully: {resp_json['proj_name']}")
        self.project = Project.from_json_resp(resp_json)
        return self.get_project(name=name)

    def get_project(self, name):
        self._login_from_conf()
        if self.username is None:
            raise Exception("Please init/login AutoNLP first")
        if self.project is None or self.project.name != name:
            resp = requests.get(url=config.HF_AUTONLP_BACKEND_API + f"/projects/{self.username}/{name}")
            logger.info(resp.json())
            proj_id = resp.json().get("id")
            if proj_id is None:
                raise Exception("Project not found, please create the project using create_project")
            else:
                resp = requests.get(
                    url=config.HF_AUTONLP_BACKEND_API + f"/projects/{self.username}/{name}",
                    headers=get_auth_headers(token=self.api_key),
                ).json()
                self.project = Project.from_json_resp(resp)
        return self.project


if __name__ == "__main__":
    client = AutoNLP()
    client.login(username="abhishek")
    project = client.create_project(name="imdb_test_4", task="binary_classification")
    # project = client.get_project(name="imdb_test_4")
    col_mapping = {"sentiment": "target", "review": "text"}

    train_files = ["/home/abhishek/datasets/imdb_folds.csv"]
    valid_files = ["/home/abhishek/datasets/imdb_valid.csv"]
    project.upload(train_files, split="train", col_mapping=col_mapping)
    project.upload(valid_files, split="valid", col_mapping=col_mapping)

    project.train()
