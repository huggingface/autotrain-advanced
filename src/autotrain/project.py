import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
from huggingface_hub import Repository
from loguru import logger
from prettytable import PrettyTable
from tqdm import tqdm

from .audio_utils import SUPPORTED_AUDIO_FILE_FORMAT, audio_file_name_iter
from .splits import TEST_SPLIT, TRAIN_SPLIT, VALID_SPLIT
from .utils import BOLD_TAG, CYAN_TAG, GREEN_TAG, PURPLE_TAG, RESET_TAG, YELLOW_TAG, get_task, http_get, http_post
from .validation import InvalidFileError, validate_file


FILE_STATUS = (
    ("☁", "Uploaded"),
    ("⌚", "Queued"),
    ("⚙", "In Progress..."),
    ("✅", "Success!"),
    ("❌", "Failed: file not found"),
    ("❌", "Failed: unsupported file type"),
    ("❌", "Failed: server error"),
    ("❌", "Invalid column mapping, please fix it and re-upload the file."),
    ("❌", "Invalid file"),
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
    ("⚙", "Preparing your models..."),
    ("❌", "Failed to download data files from the huggingface hub"),
    ("❌", "Missing 'train' or 'valid' split in data files"),
    ("❌", "Failed to process data files"),
    ("❌", "Failed to upload processed data files to the huggingface hub"),
    ("❌", "Failed to prepare your models for training"),
    ("✅", "Successfully queued your models for training!"),
)


SPLITS = (TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT)


class TrainingCancelledError(Exception):
    pass


def get_file_status(status_id: int) -> Tuple[str, str]:
    try:
        return FILE_STATUS[status_id - 1]
    except IndexError:
        return "❓", "Unhandled status! Please update autonlp"


def get_job_status(status_id: int) -> Tuple[str, str]:
    try:
        return JOB_STATUS[status_id - 1]
    except IndexError:
        return "❓", "Unhandled status! Please update autonlp"


def get_project_status(status_id: int) -> Tuple[str, str]:
    try:
        return PROJECT_STATUS[status_id - 1]
    except IndexError:
        return "❓", "Unhandled status! Please update autonlp"


def get_split(split_id: int) -> str:
    try:
        return SPLITS[split_id - 1]
    except IndexError:
        return "❓ Unhandled split! Please update autonlp"


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
        status_emoji, status = get_job_status(json_resp["status"])
        return cls(
            job_id=json_resp["id"],
            status_emoji=status_emoji,
            status=status,
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
        file_status = " ".join(get_file_status(json_resp["download_status"]))
        return cls(
            file_id=json_resp["data_file_id"],
            filename=json_resp["fname"],
            processing_status=file_status,
            split=get_split(json_resp["split"]),
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
    usd_cost: Optional[float] = None

    @classmethod
    def from_json_resp(cls, json_resp: dict, token: str):
        """Build a Project from the API response, JSON-encoded"""
        status_emoji, status = get_project_status(json_resp["status"])
        task = get_task(json_resp["task"])
        return cls(
            proj_id=json_resp["id"],
            name=json_resp["proj_name"],
            user=json_resp["username"],
            task=task,
            status_emoji=status_emoji,
            status=status,
            created_at=datetime.fromisoformat(json_resp["created_at"]),
            updated_at=datetime.fromisoformat(json_resp["updated_at"]),
            dataset_id=json_resp["dataset_id"],
            language=json_resp["config"]["language"],
            _token=token,
        )

    def refresh(self):
        """Update information about uploaded files and models attached to the project"""
        logger.info("🔄 Refreshing project status...")
        resp = http_get(path=f"/projects/{self.proj_id}", token=self._token)
        self.status_emoji, self.status = get_project_status(resp.json()["status"])

        logger.info("🔄 Refreshing uploaded files information...")
        resp = http_get(path=f"/projects/{self.proj_id}/data", token=self._token)
        json_files = resp.json()
        self.files = [UploadedFile.from_json_resp(file) for file in json_files]

        logger.info("🔄 Refreshing models information...")
        resp = http_get(path=f"/projects/{self.proj_id}/jobs", token=self._token)
        json_jobs = resp.json()
        self.training_jobs = [TrainingJob.from_json_resp(job) for job in json_jobs]

        logger.info("🔄 Refreshing cost information...")
        resp = http_get(path=f"/zeus/cost/{self.proj_id}", token=self._token)
        self.usd_cost = resp.json().get("cost_usd")

    def upload(
        self, filepaths: List[str], split: str, col_mapping: Dict[str, str], path_to_audio: Optional[str] = None
    ):
        """Uploads files to the project"""
        if self.task == "speech_recognition" and not path_to_audio:
            raise ValueError("'path_to_audio' must be provided when task is 'speech_recognition'")

        dataset_repo = self._clone_dataset_repo()
        local_dataset_dir = dataset_repo.local_dir

        for idx, file_path in enumerate(filepaths):
            if not os.path.isfile(file_path):
                logger.error(f"[{idx + 1}/{len(filepaths)}] ❌ '{file_path}' does not exist or is not a file!")
                raise FileNotFoundError(f"'{file_path}' does not exist or is not a file!")
            file_name = os.path.basename(file_path)
            file_extension = file_name.split(".")[-1]
            file_path = os.path.expanduser(file_path)

            # Validate
            logger.info(f"[{idx + 1}/{len(filepaths)}] 🔎 Validating {file_path} and column mapping...")
            validate_file(
                path=file_path,
                task=self.task,
                file_ext=file_extension,
                col_mapping=col_mapping,
            )

            # Speech recognition: check and copy audio files
            if self.task == "speech_recognition":
                dataset_repo.lfs_track(patterns=["raw/audio/*"])

                audio_dir_paths = [os.path.expanduser(path) for path in path_to_audio.split(",")]
                for audio_dir in audio_dir_paths:
                    if not os.path.isdir(audio_dir):
                        raise FileNotFoundError(f"'{audio_dir}' does not exist or is not a directory")

                audio_files_paths = []
                for audio_file_name in tqdm(
                    audio_file_name_iter(transcription_file_path=file_path, col_mapping=col_mapping),
                    desc=f"🔎 Looking for audio files in '{audio_dir_paths}'...",
                ):
                    audio_file_ext = audio_file_name.split(".")[-1]
                    if audio_file_ext not in SUPPORTED_AUDIO_FILE_FORMAT:
                        raise InvalidFileError(
                            f"Audio file '{audio_file_name}' has an unsupported extension, "
                            f"supported extensions for audio files are: {SUPPORTED_AUDIO_FILE_FORMAT}"
                        )

                    full_paths = map(lambda dirpath: os.path.join(dirpath, audio_file_name), audio_dir_paths)
                    try:
                        audio_files_paths.append(next(path for path in full_paths if os.path.isfile(path)))
                    except StopIteration as err:
                        # Not found in the provided dirs
                        raise FileNotFoundError(f"'{audio_file_name}' not found in {audio_dir_paths}") from err

                audio_dst_dir = os.path.join(local_dataset_dir, "raw", "audio")
                os.makedirs(audio_dst_dir, exist_ok=True)
                for audio_file_path in tqdm(audio_files_paths, desc=f"📦 Copying audio files to {audio_dst_dir}"):
                    audio_dst = os.path.join(audio_dst_dir, os.path.basename(audio_file_path))
                    shutil.copyfile(audio_file_path, audio_dst)

            # Copy to repo
            dst = os.path.join(local_dataset_dir, "raw", file_name)
            logger.info(f"[{idx + 1}/{len(filepaths)}] 📦 Copying {file_path} to {dst}...")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(file_path, dst)
            dataset_repo.lfs_track(patterns=[f"raw/*.{file_extension}"])

        dataset_repo.git_pull()

        try:
            logger.info("☁ Uploading files to the dataset hub...")
            dataset_repo.push_to_hub(commit_message="Upload from AutoNLP CLI")
            logger.info("✅ Successfully uploaded  the files!")
        except OSError as err:
            if "nothing to commit, working tree clean" in err.args[0]:
                logger.info("❔ Files did not change since last upload!")
                dataset_repo.git_push()
                return
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

    def train(self, noprompt=False):
        """Starts training on the models"""
        self.refresh()
        logger.info("🔎 Calculating a cost estimate for the training...")
        dataset_repo = self._clone_dataset_repo()
        local_dataset_dir = dataset_repo.local_dir
        total_number_of_lines = 0

        for file in self.files:
            with open(
                os.path.join(local_dataset_dir, "raw", file.filename), "r", encoding="utf-8", errors="ignore"
            ) as f:
                total_number_of_lines += sum(1 for line in f)

        cost_estimate = self.estimate_cost(nb_samples=total_number_of_lines)

        print(
            "\n"
            "💰 The training cost for this project will be in this range:\n"
            f" {BOLD_TAG}USD {cost_estimate['cost_min']} to USD {cost_estimate['cost_max']}{RESET_TAG}\n\n"
            " Once training is complete, we will send you an email invoice for the actual training cost within that range.\n"
        )

        if not noprompt:
            answer = input(f"Enter `{BOLD_TAG}yes{RESET_TAG}` to proceed with the training:  ")
            if answer.lower() != "yes":
                raise TrainingCancelledError

        http_get(path=f"/projects/{self.proj_id}/data/start_process", token=self._token)
        logger.info("🔥🔥 Training started!")

    def estimate_cost(self, nb_samples: int) -> Dict[str, float]:
        try:
            payload = {"num_train_samples": nb_samples}
            cost_estimate = http_post(path=f"/zeus/estimate/{self.proj_id}", token=self._token, payload=payload).json()
        except requests.exceptions.HTTPError as err:
            if err.response.status_code == 404:
                raise ValueError("❌ Unable to estimate") from err
            raise
        return cost_estimate

    def _clone_dataset_repo(self) -> Repository:
        local_dataset_dir = os.path.join(
            os.path.expanduser("~"), ".huggingface", "autonlp", "projects", self.dataset_id
        )
        if os.path.exists(local_dataset_dir):
            if os.path.isdir(os.path.join(local_dataset_dir, ".git")):
                clone_from = None
            else:
                shutil.rmtree(local_dataset_dir)
                clone_from = "https://huggingface.co/datasets/" + self.dataset_id
        else:
            clone_from = "https://huggingface.co/datasets/" + self.dataset_id
        dataset_repo = Repository(
            local_dir=local_dataset_dir,
            clone_from=clone_from,
            use_auth_token=self._token,
        )
        try:
            subprocess.run(
                "git reset --hard".split(),
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,
                encoding="utf-8",
                cwd=dataset_repo.local_dir,
            )
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)
        dataset_repo.git_pull()
        return dataset_repo

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
                f"\n💰 Project current cost: {GREEN_TAG}USD {self.usd_cost:.2f}{RESET_TAG}\n"
                if self.usd_cost is not None
                else "",
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
