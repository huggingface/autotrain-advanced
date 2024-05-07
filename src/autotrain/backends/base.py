import base64
import io
import json
import os
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Union

import requests
from huggingface_hub import HfApi
from requests.exceptions import HTTPError

from autotrain import logger
from autotrain.app_utils import run_training
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.generic.params import GenericParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.text_regression.params import TextRegressionParams
from autotrain.trainers.token_classification.params import TokenClassificationParams


@dataclass
class BaseBackend:
    params: Union[
        TextClassificationParams,
        ImageClassificationParams,
        LLMTrainingParams,
        GenericParams,
        TabularParams,
        DreamBoothTrainingParams,
        Seq2SeqParams,
        TokenClassificationParams,
        TextRegressionParams,
    ]
    backend: str

    def __post_init__(self):
        if not isinstance(self.params, GenericParams) and not self.backend.startswith("local"):
            if self.params.username is not None:
                self.username = self.params.username
            else:
                raise ValueError("Must provide username")
        else:
            self.username = self.params.username

        self.env_vars = {
            "HF_TOKEN": self.params.token,
            "AUTOTRAIN_USERNAME": self.username,
            "PROJECT_NAME": self.params.project_name,
            "TASK_ID": str(self.task_id),
            "PARAMS": json.dumps(self.params.model_dump_json()),
        }
        if isinstance(self.params, DreamBoothTrainingParams):
            self.env_vars["DATA_PATH"] = self.params.image_path
        else:
            self.env_vars["DATA_PATH"] = self.params.data_path

        if not isinstance(self.params, GenericParams):
            self.env_vars["MODEL"] = self.params.model

        if isinstance(self.params, LLMTrainingParams):
            self.task_id = 9
        elif isinstance(self.params, TextClassificationParams):
            self.task_id = 2
        elif isinstance(self.params, TabularParams):
            self.task_id = 26
        elif isinstance(self.params, GenericParams):
            self.task_id = 27
        elif isinstance(self.params, DreamBoothTrainingParams):
            self.task_id = 25
        elif isinstance(self.params, Seq2SeqParams):
            self.task_id = 28
        elif isinstance(self.params, ImageClassificationParams):
            self.task_id = 18
        elif isinstance(self.params, TokenClassificationParams):
            self.task_id = 4
        elif isinstance(self.params, TextRegressionParams):
            self.task_id = 10
        else:
            raise NotImplementedError

    def prepare(self):
        backend_id = self._create_space()
        return backend_id
