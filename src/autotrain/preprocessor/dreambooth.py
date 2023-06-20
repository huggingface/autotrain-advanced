import io
import json
from dataclasses import dataclass
from typing import Any, List

from huggingface_hub import HfApi, create_repo
from loguru import logger


@dataclass
class DreamboothPreprocessor:
    concept_images: List[Any]
    concept_name: str
    username: str
    project_name: str
    token: str

    def __post_init__(self):
        self.repo_name = f"{self.username}/autotrain-data-{self.project_name}"
        try:
            create_repo(
                repo_id=self.repo_name,
                repo_type="dataset",
                token=self.token,
                private=True,
                exist_ok=False,
            )
        except Exception:
            logger.error("Error creating repo")
            raise ValueError("Error creating repo")

    def _upload_concept_images(self, file, api):
        logger.info(f"Uploading {file} to concept1")
        api.upload_file(
            path_or_fileobj=file.name,
            path_in_repo=f"concept1/{file.name.split('/')[-1]}",
            repo_id=self.repo_name,
            repo_type="dataset",
            token=self.token,
        )

    def _upload_concept_prompts(self, api):
        _prompts = {}
        _prompts["concept1"] = self.concept_name

        prompts = json.dumps(_prompts)
        prompts = prompts.encode("utf-8")
        prompts = io.BytesIO(prompts)
        api.upload_file(
            path_or_fileobj=prompts,
            path_in_repo="prompts.json",
            repo_id=self.repo_name,
            repo_type="dataset",
            token=self.token,
        )

    def prepare(self):
        api = HfApi()
        for _file in self.concept_images:
            self._upload_concept_images(_file, api)

        self._upload_concept_prompts(api)
