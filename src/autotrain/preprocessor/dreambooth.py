import io
import json
from dataclasses import dataclass
from typing import List

from huggingface_hub import HfApi, create_repo
from loguru import logger


@dataclass
class DreamboothPreprocessor:
    num_concepts: int
    concept_images: List[List[str]]
    concept_names: List[str]
    username: str
    project_name: str
    token: str

    def __post_init__(self):
        # check if num_concepts is equal to the length of concept_images and concept_names
        if self.num_concepts != len(self.concept_images):
            raise ValueError("num_concepts is not equal to the length of concept_images")
        if self.num_concepts != len(self.concept_names):
            raise ValueError("num_concepts is not equal to the length of concept_names")

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

    def _upload_concept_images(self, file, api, concept_num):
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=f"concept{concept_num + 1}/{file.name}",
            repo_id=self.repo_name,
            repo_type="dataset",
            token=self.token,
        )

    def _upload_concept_prompts(self, api):
        _prompts = {}
        for concept_num in range(self.num_concepts):
            _prompts[f"concept{concept_num + 1}"] = self.concept_names[concept_num]

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
        logger.info(self.concept_images)
        for concept_num in range(self.num_concepts):
            for _file in self.concept_images[concept_num]:
                self._upload_concept_images(_file, api, concept_num)

        self._upload_concept_prompts(api)
