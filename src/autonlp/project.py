import requests
import os

from . import config
from loguru import logger


class Project:
    def __init__(self, proj_id, name, user):
        self.proj_id = proj_id
        self.name = name
        self.user = user

    def upload(self, files, split, col_mapping):
        jdata = {"project": self.name, "user": self.user}
        for file_path in files:
            base_name = os.path.basename(file_path)
            binary_file = open(file_path, "rb")
            files = [("files", (base_name, binary_file, "text/csv"))]
            response = requests.post(
                url=config.HF_AUTONLP_BACKEND_API + "/uploader/upload_files",
                data=jdata,
                files=files,
            )
            logger.info(response.text)

            payload = {
                "split": split,
                "col_mapping": col_mapping,
                "data_files": [{"fname": base_name, "username": self.user}],
            }
            logger.info(payload)
            response = requests.post(
                url=config.HF_AUTONLP_BACKEND_API + f"/projects/{self.proj_id}/data/add", json=payload
            )
            logger.info(response.text)

    def train(self):
        response = requests.get(url=config.HF_AUTONLP_BACKEND_API + f"/projects/{self.proj_id}/data/start_process")
        logger.info(response.text)
