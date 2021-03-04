import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from huggingface_hub import Repository
from loguru import logger
from prettytable import PrettyTable

from .splits import TEST_SPLIT, TRAIN_SPLIT, VALID_SPLIT
from .tasks import TASKS
from .utils import BOLD_TAG, CYAN_TAG, GREEN_TAG, PURPLE_TAG, RESET_TAG, YELLOW_TAG, http_get, http_post
from .validation import validate_file


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
    ("🏃", "model_training"),
    ("✅", "success"),
    ("❌", "failed"),
)

PROJECT_STATUS = (
    ("✨", "Created"),
    ("🚀", "Data processing started"),
    ("✅", "Data processing successful"),
    ("❌", "Failed to download data files from the huggingface hub"),
    ("❌", "Missing 'train' or 'valid' split in data files"),
    ("❌", "Failed to process data files"),
    ("❌", "Failed to upload processed data files to the huggingface hub"),
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
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_json_resp(cls, json_resp: dict):
        return cls(
            file_id=json_resp["data_file_id"],
            filename=json_resp["fname"],
            processing_status=FILE_STATUS[json_resp["download_status"] - 1],
            split=SPLITS[json_resp["split"] - 1],
            col_mapping=json_resp["col_mapping"],
            created_at=datetime.fromisoformat(json_resp["created_at"]),
            updated_at=datetime.fromisoformat(json_resp["updated_at"]),
        )

    def __str__(self):
        return "\n".join(
            [
                f"📁 {CYAN_TAG}{self.filename}{RESET_TAG} (id # {self.file_id})",
                f"   • {BOLD_TAG}Split{RESET_TAG}:             {self.split}",
                f"   • {BOLD_TAG}Processing status{RESET_TAG}: {self.processing_status}",
                f"   • {BOLD_TAG}Last update{RESET_TAG}:       {self.updated_at.strftime('%Y-%m-%d %H:%M Z')}",
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
    status_emoji: str
    status: str
    language: str
    created_at: datetime
    updated_at: datetime
    dataset_id: str
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
            status_emoji=PROJECT_STATUS[json_resp["status"] - 1][0],
            status=PROJECT_STATUS[json_resp["status"] - 1][1],
            created_at=datetime.fromisoformat(json_resp["created_at"]),
            updated_at=datetime.fromisoformat(json_resp["updated_at"]),
            dataset_id=json_resp["dataset_id"],
            language=json_resp["config"]["language"],
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
        local_dataset_dir = os.path.expanduser(f"~/.huggingface/autonlp/projects/{self.dataset_id}")
        if os.path.exists(local_dataset_dir):
            clone_from = None
        else:
            clone_from = "https://huggingface.co/datasets/" + self.dataset_id
        dataset_repo = Repository(
            local_dir=local_dataset_dir,
            clone_from=clone_from,
            use_auth_token=self._token,
        )
        dataset_repo.git_pull()

        for idx, file_path in enumerate(filepaths):
            if not os.path.isfile(file_path):
                logger.error(f"[{idx + 1}/{len(filepaths)}] ❌ '{file_path}' does not exist or is not a file!")
                continue
            file_name = os.path.basename(file_path)
            file_extension = file_name.split(".")[-1]
            src = os.path.expanduser(file_path)
            dst = os.path.join(local_dataset_dir, "raw", file_name)
            logger.info(f"[{idx + 1}/{len(filepaths)}] 📦 Copying {src} to {dst}...")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)

            logger.info(f"[{idx + 1}/{len(filepaths)}] 🔎 Validating {dst} and column mapping...")
            validate_file(path=dst, task=self.task, file_ext=file_extension, col_mapping=col_mapping)

            dataset_repo.lfs_track(patterns=[f"raw/**.{file_extension}"])
        try:
            logger.info("☁ Uploading files to the dataset hub...")
            dataset_repo.push_to_hub(commit_message="Upload from AutoNLP CLI")
            logger.info("✅ Successfully uploaded  the files!")
        except OSError as err:
            if "nothing to commit, working tree clean" in err.args[0]:
                logger.info("❔ Files did not change since last upload!")
                return
            else:
                logger.error("❌ Something went wrong when uploading the files!")
                raise

        for idx, file_path in enumerate(filepaths):
            file_name = os.path.basename(file_path)
            logger.info(f"[{idx + 1}/{len(filepaths)}] 📁 Registering file {file_name} into project '{file_name}'...")
            payload = {
                "split": split,
                "col_mapping": col_mapping,
                "data_files": [{"fname": file_name, "username": self.user}],
            }
            http_post(path=f"/projects/{self.proj_id}/data/add", payload=payload, token=self._token)
            logger.info(f"[{idx + 1}/{len(filepaths)}] ✅ Success!")

    def train(self):
        """Starts training on the models"""
        http_get(path=f"/projects/{self.proj_id}/data/start_process", token=self._token)
        logger.info("🔥🔥 Training started!")

    def __str__(self):
        header = "\n".join(
            [
                f"AutoNLP Project (id # {self.proj_id})",
                "~" * 35,
                f" • {BOLD_TAG}Name{RESET_TAG}:        {PURPLE_TAG}{self.name}{RESET_TAG}",
                f" • {BOLD_TAG}Owner{RESET_TAG}:       {GREEN_TAG}{self.user}{RESET_TAG}",
                f" • {BOLD_TAG}Status{RESET_TAG}:      {BOLD_TAG}{self.status_emoji} {self.status}{RESET_TAG}",
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
                descriptions = ["🤷 No files uploaded yet!"]
            else:
                sorted_files = sorted(self.files, key=lambda file: file.split)  # Sort by split
                descriptions = [str(file) for file in sorted_files]
        printout.append(
            "\n".join(
                [
                    "~" * 14 + f" {BOLD_TAG}Files{RESET_TAG} " + "~" * 14,
                    "",
                    "Dataset ID:",
                    f"{CYAN_TAG}{self.dataset_id}{RESET_TAG}",
                    "",
                ]
                + descriptions
            )
        )

        # Training jobs information
        if self.training_jobs is None:
            jobs_str = "❓ Models information unknown, update the project"
        else:
            if len(self.training_jobs) == 0:
                jobs_str = "🤷 No train jobs started yet!"
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
