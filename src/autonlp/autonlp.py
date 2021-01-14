# flake8: noqa
# coding=utf-8
# Copyright 2020 The HuggingFace Team
# Lint as: python3
# pylint: enable=line-too-long

import json
import requests
from typing import Union, List
import os
import requests
from . import config
from .tasks import TASKS
from .project import Project
from loguru import logger


class AutoNLP:
    def __init__(self, username: str) -> None:
        self.org = "huggingface"
        self.username = username
        self.project_id = -1

    def login(self):
        # os.makedirs()
        pass

    def create_project(self, name: str, task: str):
        task_id = TASKS.get(task, -1)
        if task_id == -1:
            raise Exception(f"Invalid task specified. Please choose one of {list(TASKS.keys())}")
        payload = {
            "username": self.username,
            "org": self.org,
            "proj_name": name,
            "task": task_id,
            "config": {"version": 0, "patch": 1},
        }
        try:
            resp = requests.post(url=config.HF_AUTONLP_BACKEND_API + "/projects/", json=payload)
        except requests.exceptions.ConnectionError:
            raise Exception("API is currently not available")
        resp_json = resp.json()
        if resp_json["created"] is True:
            logger.info(f"Created project: {resp_json['proj_name']}")
        else:
            logger.info(f"Project already exists. Loaded successfully: {resp_json['proj_name']}")
        self.project_id = resp_json["id"]
        return self.get_project(name=name)

    def get_project(self, name):
        if self.org is None or self.username is None:
            raise Exception("Please init the AutoNLP class first")
        if self.project_id == -1:
            resp = requests.get(url=config.HF_AUTONLP_BACKEND_API + f"/projects/{self.org}/{self.username}/{name}")
            proj_id = resp.get("id")
            if proj_id is None:
                raise Exception("Project not found, please create the project using create_project")
        return Project(proj_id=self.project_id, name=name, org=self.org, user=self.username)


if __name__ == "__main__":
    client = AutoNLP(username="abhishek")
    project = client.create_project(name="imdb_test_4", task="binary_classification")
    # project = client.get_project(name="imdb_test_4")
    col_mapping = {"sentiment": "target", "review": "text"}

    train_files = ["/home/abhishek/datasets/imdb_folds.csv"]
    valid_files = ["/home/abhishek/datasets/imdb_valid.csv"]
    project.upload(train_files, split="train", col_mapping=col_mapping)
    project.upload(valid_files, split="valid", col_mapping=col_mapping)

    project.train()
