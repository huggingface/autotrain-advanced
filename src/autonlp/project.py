import os

from dataclasses import dataclass
from datetime import datetime
import requests
from loguru import logger
from typing import Optional, List, Dict

from . import config
from .tasks import TASKS
from .config import get_auth_headers


@dataclass
class Project:
    proj_id: int
    name: str
    user: str
    task: str
    status: str
    created_at: datetime
    updated_at: datetime
    files: Optional[List] = None
    training_jobs: Optional[List] = None

    @classmethod
    def from_json_resp(cls, json_resp: dict):
        return cls(
            proj_id=json_resp["id"],
            name=json_resp["proj_name"],
            user=json_resp["username"],
            task=list(filter(lambda key: TASKS[key] == json_resp["task_id"], TASKS.keys())),
            status="ACTIVE" if json_resp["status"] == 1 else "INACTIVE",
            created_at=datetime.fromisoformat(json_resp["created_at"]),
            updated_at=datetime.fromisoformat(json_resp["updated_at"]),
        )

    def __str__(self):
        header = "\n".join(
            [
                f"AutoNLP Project (id # {self.proj_id}) - {self.status.upper()}",
                "~" * 35,
                f" - Name:        {self.name}",
                f" - Owner:       {self.user}",
                f" - Task:        {self.task.title().replace('_', ' ')}",
                f" - Created at:  {self.created_at.strftime('%Y-%m-%d %H:%M Z')}"
                f" - Last update: {self.updated_at.strftime('%Y-%m-%d %H:%M Z')}",
            ]
        )
        printout = [header]
        return "\n".join(printout)

    def upload(self, files: List[str], split: str, col_mapping: Dict[str, str], token: str):
        jdata = {"project": self.name, "username": self.user}
        for file_path in files:
            base_name = os.path.basename(file_path)
            binary_file = open(file_path, "rb")
            files = [("files", (base_name, binary_file, "text/csv"))]
            response = requests.post(
                url=config.HF_AUTONLP_BACKEND_API + "/uploader/upload_files",
                data=jdata,
                files=files,
                headers=get_auth_headers(token),
            )
            logger.info(response.text)

            payload = {
                "split": split,
                "col_mapping": col_mapping,
                "data_files": [{"fname": base_name, "username": self.user}],
            }
            logger.info(payload)
            response = requests.post(
                url=config.HF_AUTONLP_BACKEND_API + f"/projects/{self.proj_id}/data/add",
                json=payload,
                headers=get_auth_headers(token),
            )
            logger.info(response.text)

    def train(self, token: str):
        response = requests.get(
            url=config.HF_AUTONLP_BACKEND_API + f"/projects/{self.proj_id}/data/start_process",
            headers=get_auth_headers(token),
        )
        logger.info(response.text)
