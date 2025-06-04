import os
from dataclasses import dataclass
from typing import Optional

import requests

from autotrain import logger


AUTOTRAIN_API = os.environ.get("AUTOTRAIN_API", "https://autotrain-projects-autotrain-advanced.hf.space/")

BACKENDS = {
    "spaces-a10g-large": "a10g-large",
    "spaces-a10g-small": "a10g-small",
    "spaces-a100-large": "a100-large",
    "spaces-t4-medium": "t4-medium",
    "spaces-t4-small": "t4-small",
    "spaces-cpu-upgrade": "cpu-upgrade",
    "spaces-cpu-basic": "cpu-basic",
    "spaces-l4x1": "l4x1",
    "spaces-l4x4": "l4x4",
    "spaces-l40sx1": "l40sx1",
    "spaces-l40sx4": "l40sx4",
    "spaces-l40sx8": "l40sx8",
    "spaces-a10g-largex2": "a10g-largex2",
    "spaces-a10g-largex4": "a10g-largex4",
}


PARAMS = {}
PARAMS["llm"] = {
    "target_modules": "all-linear",
    "log": "tensorboard",
    "mixed_precision": "fp16",
    "quantization": "int4",
    "peft": True,
    "block_size": 1024,
    "epochs": 3,
    "padding": "right",
    "chat_template": "none",
    "max_completion_length": 128,
    "distributed_backend": "ddp",
    "scheduler": "linear",
    "merge_adapter": True,
}

PARAMS["text-classification"] = {
    "mixed_precision": "fp16",
    "log": "tensorboard",
}

PARAMS["st"] = {
    "mixed_precision": "fp16",
    "log": "tensorboard",
}

PARAMS["image-classification"] = {
    "mixed_precision": "fp16",
    "log": "tensorboard",
}

PARAMS["image-object-detection"] = {
    "mixed_precision": "fp16",
    "log": "tensorboard",
}

PARAMS["seq2seq"] = {
    "mixed_precision": "fp16",
    "target_modules": "all-linear",
    "log": "tensorboard",
}

PARAMS["tabular"] = {
    "categorical_imputer": "most_frequent",
    "numerical_imputer": "median",
    "numeric_scaler": "robust",
}

PARAMS["token-classification"] = {
    "mixed_precision": "fp16",
    "log": "tensorboard",
}

PARAMS["text-regression"] = {
    "mixed_precision": "fp16",
    "log": "tensorboard",
}

PARAMS["image-regression"] = {
    "mixed_precision": "fp16",
    "log": "tensorboard",
}

PARAMS["vlm"] = {
    "mixed_precision": "fp16",
    "target_modules": "all-linear",
    "log": "tensorboard",
    "quantization": "int4",
    "peft": True,
    "epochs": 3,
}

PARAMS["extractive-qa"] = {
    "mixed_precision": "fp16",
    "log": "tensorboard",
    "max_seq_length": 512,
    "max_doc_stride": 128,
}

PARAMS["automatic-speech-recognition"] = {
    "mixed_precision": "fp16",
    "log": "tensorboard",
    "max_duration": 30.0,
    "sampling_rate": 16000,
    "audio_column": "audio",
    "text_column": "transcription",
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.01,
    "eval_strategy": "epoch",
    "save_total_limit": 1,
    "auto_find_batch_size": False,
    "logging_steps": -1,
}

DEFAULT_COLUMN_MAPPING = {
    "text-classification": {"text_column": "text", "target_column": "label"},
    "seq2seq": {"text_column": "text", "target_column": "target"},
    "text-regression": {"text_column": "text", "target_column": "target"},
    "token-classification": {"text_column": "tokens", "target_column": "tags"},
    "image-classification": {"image_column": "image", "target_column": "label"},
    "image-regression": {"image_column": "image", "target_column": "target"},
    "image-object-detection": {"image_column": "image", "objects_column": "objects"},
    "tabular:classification": {"id_column": "id", "target__columns": ["target"]},
    "tabular:regression": {"id_column": "id", "target_columns": ["target"]},
    "extractive-qa": {
        "text_column": "context",
        "question_column": "question",
        "answer_column": "answers",
    },
    "automatic-speech-recognition": {
        "audio_column": "audio",
        "text_column": "transcription",
    },
}

VALID_TASKS = [k for k in DEFAULT_COLUMN_MAPPING.keys()]


@dataclass
class Client:
    """
    A client to interact with the AutoTrain API.
    Attributes:
        host (Optional[str]): The host URL for the AutoTrain API.
        token (Optional[str]): The authentication token for the API.
        username (Optional[str]): The username for the API.
    Methods:
        __post_init__():
            Initializes the client with default values if not provided and sets up headers.
        __str__():
            Returns a string representation of the client with masked token.
        __repr__():
            Returns a string representation of the client with masked token.
        create(project_name: str, task: str, base_model: str, hardware: str, dataset: str, train_split: str, column_mapping: Optional[dict] = None, params: Optional[dict] = None, valid_split: Optional[str] = None):
            Creates a new project on the AutoTrain platform.
        get_logs(job_id: str):
            Retrieves logs for a given job ID.
        stop_training(job_id: str):
            Stops the training for a given job ID.
        _get_task_specific_params():
            Get task-specific parameters.
    """

    host: Optional[str] = None
    token: Optional[str] = None
    username: Optional[str] = None

    def __post_init__(self):
        if self.host is None:
            self.host = AUTOTRAIN_API

        if self.token is None:
            self.token = os.environ.get("HF_TOKEN")

        if self.username is None:
            self.username = os.environ.get("HF_USERNAME")

        if self.token is None or self.username is None:
            raise ValueError("Please provide a valid username and token")

        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def __str__(self):
        return f"Client(host={self.host}, token=****, username={self.username})"

    def __repr__(self):
        return self.__str__()

    def create(
        self,
        project_name: str,
        task: str,
        base_model: str,
        backend: str,
        dataset: str,
        train_split: str,
        column_mapping: Optional[dict] = None,
        params: Optional[dict] = None,
        valid_split: Optional[str] = None,
    ):

        if task not in VALID_TASKS:
            raise ValueError(f"Invalid task. Valid tasks are: {VALID_TASKS}")

        if backend not in BACKENDS:
            raise ValueError(f"Invalid backend. Valid backends are: {list(BACKENDS.keys())}")

        url = f"{self.host}/api/create_project"

        if task == "llm:defaut":
            task = "llm:generic"

        if params is None:
            params = {}

        if task.startswith("llm"):
            params = {k: v for k, v in PARAMS["llm"].items() if k not in params}
        elif task.startswith("st"):
            params = {k: v for k, v in PARAMS["st"].items() if k not in params}
        else:
            params = {k: v for k, v in PARAMS[task].items() if k not in params}

        if column_mapping is None:
            column_mapping = DEFAULT_COLUMN_MAPPING[task]

        # check if column_mapping is valid for the task
        default_col_map = DEFAULT_COLUMN_MAPPING[task]
        missing_cols = []
        for k, _ in default_col_map.items():
            if k not in column_mapping.keys():
                missing_cols.append(k)

        if missing_cols:
            raise ValueError(f"Missing columns in column_mapping: {missing_cols}")

        data = {
            "project_name": project_name,
            "task": task,
            "base_model": base_model,
            "hardware": backend,
            "params": params,
            "username": self.username,
            "column_mapping": column_mapping,
            "hub_dataset": dataset,
            "train_split": train_split,
            "valid_split": valid_split,
        }
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 200:
            resp = response.json()
            logger.info(
                f"Project created successfully. Job ID: {resp['job_id']}. View logs at: https://hf.co/spaces/{resp['job_id']}"
            )
            return resp
        else:
            logger.error(f"Error creating project: {response.json()}")
            return response.json()

    def get_logs(self, job_id: str):
        url = f"{self.host}/api/logs"
        data = {"jid": job_id}
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()

    def stop_training(self, job_id: str):
        url = f"{self.host}/api/stop_training/{job_id}"
        data = {"jid": job_id}
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()

    def _get_task_specific_params(self):
        """Get task-specific parameters."""
        if self.task == "automatic-speech-recognition":
            return {
                "max_duration": 30.0,
                "sampling_rate": 16000,
                "audio_column": "audio",
                "text_column": "transcription",
                "max_grad_norm": 1.0,
                "weight_decay": 0.01,
                "warmup_ratio": 0.1,
                "early_stopping_patience": 3,
                "early_stopping_threshold": 0.01,
                "eval_strategy": "epoch",
                "save_total_limit": 1,
                "auto_find_batch_size": False,
                "logging_steps": -1,
            }
        return {}
