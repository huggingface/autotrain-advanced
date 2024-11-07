import os
from dataclasses import dataclass
from typing import Optional

import requests


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

PARAMS["dreambooth"] = {
    "vae_model": "",
    "num_steps": 500,
    "disable_gradient_checkpointing": False,
    "mixed_precision": "fp16",
    "batch_size": 1,
    "gradient_accumulation": 4,
    "resolution": 1024,
    "use_8bit_adam": False,
    "xformers": False,
    "train_text_encoder": False,
    "lr": 1e-4,
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

DEFAULT_COLUMN_MAPPING = {}
DEFAULT_COLUMN_MAPPING["llm:sft"] = {"text": "text"}
DEFAULT_COLUMN_MAPPING["llm:generic"] = {"text": "text"}
DEFAULT_COLUMN_MAPPING["llm:default"] = {"text": "text"}
DEFAULT_COLUMN_MAPPING["llm:dpo"] = {"prompt": "prompt", "text": "chosen", "rejected_text": "rejected"}
DEFAULT_COLUMN_MAPPING["llm:orpo"] = {"prompt": "prompt", "text": "chosen", "rejected_text": "rejected"}
DEFAULT_COLUMN_MAPPING["llm:reward"] = {"text": "chosen", "rejected_text": "rejected"}
DEFAULT_COLUMN_MAPPING["vlm:captioning"] = {"image": "image", "text": "caption"}
DEFAULT_COLUMN_MAPPING["vlm:vqa"] = {"image": "image", "prompt": "question", "text": "answer"}
DEFAULT_COLUMN_MAPPING["st:pair"] = {"sentence1": "anchor", "sentence2": "positive"}
DEFAULT_COLUMN_MAPPING["st:pair_class"] = {"sentence1": "premise", "sentence2": "hypothesis", "target": "label"}
DEFAULT_COLUMN_MAPPING["st:pair_score"] = {"sentence1": "sentence1", "sentence2": "sentence2", "target": "score"}
DEFAULT_COLUMN_MAPPING["st:triplet"] = {"sentence1": "anchor", "sentence2": "positive", "sentence3": "negative"}
DEFAULT_COLUMN_MAPPING["st:qa"] = {"sentence1": "query", "sentence2": "answer"}
DEFAULT_COLUMN_MAPPING["text-classification"] = {"text": "text", "label": "target"}
DEFAULT_COLUMN_MAPPING["seq2seq"] = {"text": "text", "label": "target"}
DEFAULT_COLUMN_MAPPING["text-regression"] = {"text": "text", "label": "target"}
DEFAULT_COLUMN_MAPPING["token-classification"] = {"text": "tokens", "label": "tags"}
DEFAULT_COLUMN_MAPPING["dreambooth"] = {"image": "image"}
DEFAULT_COLUMN_MAPPING["image-classification"] = {"image": "image", "label": "label"}
DEFAULT_COLUMN_MAPPING["image-regression"] = {"image": "image", "label": "target"}
DEFAULT_COLUMN_MAPPING["image-object-detection"] = {"image": "image", "objects": "objects"}
DEFAULT_COLUMN_MAPPING["tabular:classification"] = {"id": "id", "label": "target"}
DEFAULT_COLUMN_MAPPING["tabular:regression"] = {"id": "id", "label": "target"}
DEFAULT_COLUMN_MAPPING["extractive-qa"] = {"text": "context", "question": "question", "answer": "answers"}

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
    """

    host: Optional[str] = None
    token: Optional[str] = None
    username: Optional[str] = None

    def __post_init__(self):
        if self.host is None:
            self.host = "https://autotrain-projects-autotrain-advanced.hf.space/"

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
        hardware: str,
        dataset: str,
        train_split: str,
        column_mapping: Optional[dict] = None,
        params: Optional[dict] = None,
        valid_split: Optional[str] = None,
    ):

        if task not in VALID_TASKS:
            raise ValueError(f"Invalid task. Valid tasks are: {VALID_TASKS}")

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

        if task == "dreambooth" and params.get("prompt") is None:
            raise ValueError("Please provide a prompt for the DreamBooth task")

        data = {
            "project_name": project_name,
            "task": task,
            "base_model": base_model,
            "hardware": hardware,
            "params": params,
            "username": self.username,
            "column_mapping": column_mapping,
            "hub_dataset": dataset,
            "train_split": train_split,
            "valid_split": valid_split,
        }
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()

    def get_logs(self, job_id: str):
        url = f"{self.host}/api/logs/{job_id}"
        response = requests.get(url, headers=self.headers)
        return response.json()

    def stop_training(self, job_id: str):
        url = f"{self.host}/api/stop_training/{job_id}"
        response = requests.get(url, headers=self.headers)
        return response.json()
