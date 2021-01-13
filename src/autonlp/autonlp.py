# flake8: noqa
# coding=utf-8
# Copyright 2020 The HuggingFace Team
# Lint as: python3
# pylint: enable=line-too-long

import json
import requests
from typing import Union, List

import requests
from . import config
from .tasks import TASKS
from .project import Project


class AutoNLP:
    def __init__(self, org: str, username: str) -> None:
        self.org = org
        self.username = username

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
        return resp_json

    def get_project(self, name):
        if self.org is None or self.username is None:
            raise Exception("Please init the AutoNLP class first")
        return Project(name=name, org=self.org, user=self.username)


if __name__ == "__main__":
    client = AutoNLP(org="huggingface", username="abhishek")
    resp = client.create_project(name="imdb_test_1", task="binary_classification")

    col_mapping = {"sentiment": "target", "review": "text"}
    project = client.get_project(name="imdb_test_1")
    train_files = ["/home/abhishek/datasets/imdb_folds.csv"]
    valid_files = ["/home/abhishek/datasets/imdb_valid.csv"]
    project.upload(train_files, split="train", col_mapping=col_mapping)
    project.upload(valid_files, split="valid", col_mapping=col_mapping)
    project.train()
