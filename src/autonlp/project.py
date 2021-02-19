import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import requests
from loguru import logger
from prettytable import PrettyTable

from .splits import TEST_SPLIT, TRAIN_SPLIT, VALID_SPLIT
from .tasks import TASKS
from .utils import BOLD_TAG, CYAN_TAG, GREEN_TAG, PURPLE_TAG, RESET_TAG, YELLOW_TAG, http_get, http_post


FILE_STATUS = (
    "☁ Uploaded",
    "⌚ Queued",
    "⚙ In Progress...",
    "✅ Success!",
    "❌ Failed: file not found",
    "❌ Failed: unsupported file type",
    "❌ Failed: server error",
    "❌ Invalid column mapping, please fix it and re-upload the file.",
)

JOB_STATUS = (
    ("⌚", "queued"),
    ("🚀", "start"),
    ("⚙", "data_munging"),
    ("🏃‍♂️", "model_training"),
    ("✅", "success"),
    ("❌", "failed"),
)

SPLITS = (TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT)


@dataclass
class TrainingJob:
    """A training job in AutoNLP"""

    job_id: int
    status: str
    status_emoji: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_json_resp(cls, json_resp: dict):
        return cls(
            job_id=json_resp["id"],
            status_emoji=JOB_STATUS[json_resp["status"] - 1][0],
            status=JOB_STATUS[json_resp["status"] - 1][1],
            created_at=datetime.fromisoformat(json_resp["created_at"]),
            updated_at=datetime.fromisoformat(json_resp["updated_at"]),
        )

    def __str__(self):
        return "\n".join(
            [
                f"📚 Model # {self.job_id}",
                f"   • {BOLD_TAG}Status{RESET_TAG}:      {self.status_emoji} {self.status}",
                f"   • {BOLD_TAG}Created at{RESET_TAG}:  {self.created_at.strftime('%Y-%m-%d %H:%M Z')}",
                f"   • {BOLD_TAG}Last update{RESET_TAG}: {self.updated_at.strftime('%Y-%m-%d %H:%M Z')}",
            ]
        )


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
            processing_status=FILE_STATUS[json_resp["download_status"] - 1],
            split=SPLITS[json_resp["split"] - 1],
            col_mapping=json_resp["col_mapping"],
        )

    def __str__(self):
        return "\n".join(
            [
                f"📁 {CYAN_TAG}{self.filename}{RESET_TAG} (id # {self.file_id})",
                f"   • {BOLD_TAG}Split{RESET_TAG}:             {self.split}",
                f"   • {BOLD_TAG}Processing status{RESET_TAG}: {self.processing_status}",
            ]
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
        """Update information about uploaded files and models attached to the project"""
        logger.info("🔄 Refreshing uploaded files information...")
        resp = http_get(path=f"/projects/{self.proj_id}/data", token=self._token)
        json_files = resp.json()
        self.files = [UploadedFile.from_json_resp(file) for file in json_files]

        logger.info("🔄 Refreshing models information...")
        resp = http_get(path=f"/projects/{self.proj_id}/jobs", token=self._token)
        json_jobs = resp.json()
        self.training_jobs = [TrainingJob.from_json_resp(job) for job in json_jobs]

    def upload(self, filepaths: List[str], split: str, col_mapping: Dict[str, str]):
        """Uploads files to the project"""
        response = http_get(path=f"/projects/{self.proj_id}/data/upload_url", token=self._token).json()
        upload_url = response["url"]
        upload_data = response["fields"]

        for idx, file_path in enumerate(filepaths):
            file_name = os.path.basename(file_path)
            logger.info(f"☁ Uploading {file_name} ... [{idx + 1}/{len(filepaths)}]")
            with open(file_path, "rb") as f:
                files = {"file": (file_name, f)}
                try:
                    upload_repsonse = requests.post(url=upload_url, data=upload_data, files=files)
                except requests.ConnectionError:
                    logger.error(f"❌ Connection error while uploading, {file_name} has not been uploaded.")
                    continue

            if upload_repsonse.status_code != 204:
                logger.error(
                    f"❌ An error occurred while uploading, {file_name} has not been uploaded. Please report the following to the HuggingFace team:"
                )
                logger.error(f"HTTP status code: {upload_repsonse.status_code}")
                logger.error(upload_repsonse.text)
                continue

            logger.info(f"✅ Successfully uploaded {file_name}! Adding the file to project: {self.name}")
            payload = {
                "split": split,
                "col_mapping": col_mapping,
                "data_files": [{"fname": file_name, "username": self.user}],
            }
            http_post(path=f"/projects/{self.proj_id}/data/add", payload=payload, token=self._token)

        logger.info(f"✅ Successfully uploaded {len(filepaths)} files to AutoNLP!")

    def train(self):
        """Starts training on the models"""
        http_get(path=f"/projects/{self.proj_id}/data/start_process", token=self._token)
        logger.info("🔥🔥 Training started!")

    def __str__(self):
        header = "\n".join(
            [
                f"AutoNLP Project (id # {self.proj_id}) - {self.status.upper()}",
                "",
                "~" * 35,
                f" • {BOLD_TAG}Name{RESET_TAG}:        {PURPLE_TAG}{self.name}{RESET_TAG}",
                f" • {BOLD_TAG}Owner{RESET_TAG}:       {GREEN_TAG}{self.user}{RESET_TAG}",
                f" • {BOLD_TAG}Task{RESET_TAG}:        {YELLOW_TAG}{self.task.title().replace('_', ' ')}{RESET_TAG}",
                f" • {BOLD_TAG}Created at{RESET_TAG}:  {self.created_at.strftime('%Y-%m-%d %H:%M Z')}",
                f" • {BOLD_TAG}Last update{RESET_TAG}: {self.updated_at.strftime('%Y-%m-%d %H:%M Z')}",
                "",
            ]
        )
        printout = [header]

        # Uploaded files information
        if self.files is None:
            descriptions = ["❓ Files information unknown, update the project"]
        else:
            if len(self.files) == 0:
                descriptions = ["🤷‍♂ No files uploaded yet!"]
            else:
                sorted_files = sorted(self.files, key=lambda file: file.split)  # Sort by split
                descriptions = [str(file) for file in sorted_files]
        printout.append("\n".join(["~" * 14 + f" {BOLD_TAG}Files{RESET_TAG} " + "~" * 14, ""] + descriptions))

        # Training jobs information
        if self.training_jobs is None:
            jobs_str = "❓ Models information unknown, update the project"
        else:
            if len(self.training_jobs) == 0:
                jobs_str = "🤷‍♂ No train jobs started yet!"
            else:
                model_table = PrettyTable(["", "ID", "Status", "Creation date", "Last update"])
                for job in sorted(self.training_jobs, key=lambda job: job.job_id):
                    model_table.add_row(
                        [
                            job.status_emoji,
                            job.job_id,
                            job.status,
                            job.created_at.strftime("%Y-%m-%d %H:%M Z"),
                            job.updated_at.strftime("%Y-%m-%d %H:%M Z"),
                        ]
                    )
                jobs_str = str(model_table)
        printout.append("\n".join(["", "~" * 12 + f" {BOLD_TAG}Models{RESET_TAG} " + "~" * 11, "", jobs_str]))

        return "\n".join(printout)
