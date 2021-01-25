import os

import requests
from loguru import logger

from . import config


class Project:
    def __init__(self, proj_id, name, user):
        self.proj_id = proj_id
        self.name = name
        self.user = user
        self.infos = None
        self.data_files = []
        self.training_jobs = []

    def fetch_infos(self, force_fetch=False):
        if force_fetch is False and self.infos is not None:
            return
        try:
            resp = requests.get(url=config.HF_AUTONLP_BACKEND_API + "/projects/" + self.proj_id)
        except requests.exceptions.ConnectionError:
            raise Exception("API is currently not available")
        self.infos = resp.json()

    def fetch_data_files(self, force_fetch=False):
        if force_fetch is False and self.data_files is not None:
            return
        try:
            resp = requests.get(url=config.HF_AUTONLP_BACKEND_API + "/projects/" + self.proj_id + "/data/")
        except requests.exceptions.ConnectionError:
            raise Exception("API is currently not available")
        self.data_files = resp.json()

    def fetch_data_files(self, force_fetch=False):
        raise NotImplementedError

    def upload(self, files, split, col_mapping):
        jdata = {"project": self.name, "username": self.user}
        for file_path in files:
            base_name = os.path.basename(file_path)
            binary_file = open(file_path, "rb")
            files = [("files", (base_name, binary_file, "text/csv"))]
            response = requests.post(
                url=config.HF_AUTONLP_BACKEND_API + "/uploader/upload_files", data=jdata, files=files,
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
