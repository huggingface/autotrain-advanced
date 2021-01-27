import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import requests
from loguru import logger
from tqdm import tqdm

from . import config
from .splits import TEST_SPLIT, TRAIN_SPLIT, VALID_SPLIT
from .tasks import TASKS
from .utils import (
    BOLD_TAG,
    CYAN_TAG,
    GREEN_TAG,
    PURPLE_TAG,
    RED_TAG,
    RESET_TAG,
    YELLOW_TAG,
    http_get,
    http_post,
    http_upload_files,
)


STATUS = (
    "☁ Uploaded",
    "⌚ Queued",
    "⚙ In Progress...",
    "✅ Success!",
    "❌ Failed: file not found",
    "❌ Failed: unsupported file type",
    "❌ Failed: server error",
)

SPLITS = (TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT)


@dataclass
class UploadedFile:
    """A file uploaded to an AutoNLP project"""

    file_id: int
    filename: str
    processing_status: str
    split: str
    col_mapping: Dict[str, str]

    @classmethod
    def from_json_resp(cls, json_resp: dict):
        return cls(
            file_id=json_resp["data_file_id"],
            filename=json_resp["fname"],
            processing_status=STATUS[json_resp["download_status"] - 1],
            split=SPLITS[json_resp["split"] - 1],
            col_mapping=json_resp["col_mapping"],
        )


@dataclass
class Project:
    """An AutoNLP project"""

    _token: str
    proj_id: int
    name: str
    user: str
    task: str
    status: str
    config: Dict[str, str]
    created_at: datetime
    updated_at: datetime
    files: Optional[List[UploadedFile]] = None
    training_jobs: Optional[List] = None

    @classmethod
    def from_json_resp(cls, json_resp: dict, token: str):
        """Build a Project from the API response, JSON-encoded"""
        return cls(
            proj_id=json_resp["id"],
            name=json_resp["proj_name"],
            user=json_resp["username"],
            task=list(filter(lambda key: TASKS[key] == json_resp["task"], TASKS.keys()))[0],
            config=json_resp["config"],
            status="ACTIVE" if json_resp["status"] == 1 else "INACTIVE",
            created_at=datetime.fromisoformat(json_resp["created_at"]),
            updated_at=datetime.fromisoformat(json_resp["updated_at"]),
            _token=token,
        )

    def refresh(self):
        """Update information about uploaded files and training jobs attached to the project"""
        logger.info("🔄 Refreshing uploaded files information...")
        resp = http_get(path=f"/projects/{self.proj_id}/data/", token=self._token)
        json_files = resp.json()
        self.files = [UploadedFile.from_json_resp(file) for file in json_files]

    def upload(self, filepaths: List[str], split: str, col_mapping: Dict[str, str]):
        """Uploads files to the project"""
        jdata = {"project": self.name, "username": self.user}
        for file_path in tqdm(filepaths, desc="Uploaded files"):
            base_name = os.path.basename(file_path)
            binary_file = open(file_path, "rb")
            files = [("files", (base_name, binary_file, "text/csv"))]
            response = http_upload_files(
                path="/uploader/upload_files", data=jdata, files_info=files, token=self._token
            )
            payload = {
                "split": split,
                "col_mapping": col_mapping,
                "data_files": [{"fname": base_name, "username": self.user}],
            }
            response = http_post(path=f"/projects/{self.proj_id}/data/add", payload=payload, token=self._token)
        logger.info(f"✅ Successfully uploaded {len(filepaths)} files to AutoNLP!")

    def train(self):
        """Starts training on the models"""
        response = http_get(path=f"/projects/{self.proj_id}/data/start_process", token=self._token)
        logger.info("🔥🔥 Training started!")

    def __str__(self):
        header = "\n".join(
            [
                f"AutoNLP Project (id # {self.proj_id}) - {self.status.upper()}",
                "",
                "~" * 35,
                f" - {BOLD_TAG}Name{RESET_TAG}:        {PURPLE_TAG}{self.name}{RESET_TAG}",
                f" - {BOLD_TAG}Owner{RESET_TAG}:       {GREEN_TAG}{self.user}{RESET_TAG}",
                f" - {BOLD_TAG}Task{RESET_TAG}:        {YELLOW_TAG}{self.task.title().replace('_', ' ')}{RESET_TAG}",
                f" - {BOLD_TAG}Created at{RESET_TAG}:  {self.created_at.strftime('%Y-%m-%d %H:%M Z')}",
                f" - {BOLD_TAG}Last update{RESET_TAG}: {self.updated_at.strftime('%Y-%m-%d %H:%M Z')}",
                "",
            ]
        )
        printout = [header]
        if self.files is not None:
            files = sorted(self.files, key=lambda file: file.split)
            descriptions = [
                "\n".join(
                    [
                        f"📁 {CYAN_TAG}{file.filename}{RESET_TAG} (id # {file.file_id})",
                        f"   > {BOLD_TAG}Split{RESET_TAG}:             {file.split}",
                        f"   > {BOLD_TAG}Processing status{RESET_TAG}: {file.processing_status}",
                    ]
                )
                for file in self.files
            ]
            printout.append("\n".join(["~" * 14 + f" {BOLD_TAG}Files{RESET_TAG} " + "~" * 14, ""] + descriptions))
        return "\n".join(printout)
