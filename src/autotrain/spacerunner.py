import io
import json
import os
from dataclasses import dataclass
from typing import Union

from huggingface_hub import HfApi

from autotrain.dataset import AutoTrainDataset
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.text_classification.params import TextClassificationParams


@dataclass
class SpaceRunner:
    params: Union[TextClassificationParams, ImageClassificationParams, LLMTrainingParams]
    backend: str

    def __post_init__(self):
        self.spaces_backends = {
            "a10gl": "a10g-large",
            "a10gs": "a10g-small",
            "a100": "a100-large",
            "t4m": "t4-medium",
            "t4s": "t4-small",
        }
        self.username = self.params.repo_id.split("/")[0]

    def prepare(self):
        if isinstance(self.params, LLMTrainingParams):
            self.task_id = 9
            data_path = self._llm_munge_data()
            self.params.data_path = data_path
            space_id = self._create_space()
            return space_id
        raise NotImplementedError

    def _create_readme(self):
        _readme = "---\n"
        _readme += f"title: {self.params.project_name}\n"
        _readme += "emoji: 🚀\n"
        _readme += "colorFrom: green\n"
        _readme += "colorTo: indigo\n"
        _readme += "sdk: docker\n"
        _readme += "pinned: false\n"
        _readme += "duplicated_from: autotrain-projects/autotrain-advanced\n"
        _readme += "---\n"
        _readme = io.BytesIO(_readme.encode())
        return _readme

    def _add_secrets(self, api, repo_id):
        api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value=self.params.token)
        api.add_space_secret(repo_id=repo_id, key="AUTOTRAIN_USERNAME", value=self.username)
        api.add_space_secret(repo_id=repo_id, key="PROJECT_NAME", value=self.params.project_name)
        api.add_space_secret(repo_id=repo_id, key="PARAMS", value=json.dumps(self.params.json()))
        api.add_space_secret(repo_id=repo_id, key="DATA_PATH", value=self.params.data_path)
        api.add_space_secret(repo_id=repo_id, key="TASK_ID", value=str(self.task_id))
        api.add_space_secret(repo_id=repo_id, key="MODEL", value=self.params.model)
        api.add_space_secret(repo_id=repo_id, key="OUTPUT_MODEL_REPO", value=self.params.repo_id)

    def _create_space(self):
        api = HfApi(token=self.params.token)
        repo_id = f"{self.username}/autotrain-{self.params.project_name}"
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            space_hardware=self.spaces_backends[self.backend.split("-")[1].lower()],
            private=True,
        )
        self._add_secrets(api, repo_id)
        readme = self._create_readme()
        api.upload_file(
            path_or_fileobj=readme,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="space",
        )

        _dockerfile = "FROM huggingface/autotrain-advanced:latest\nCMD autotrain api --port 7860 --host 0.0.0.0"
        _dockerfile = io.BytesIO(_dockerfile.encode())
        api.upload_file(
            path_or_fileobj=_dockerfile,
            path_in_repo="Dockerfile",
            repo_id=repo_id,
            repo_type="space",
        )
        return repo_id

    def _llm_munge_data(self):
        train_data_path = f"{self.params.data_path}/{self.params.train_split}.csv"
        if self.params.valid_split is not None:
            valid_data_path = f"{self.params.data_path}/{self.params.valid_split}.csv"
        else:
            valid_data_path = []
        if os.path.exists(train_data_path):
            dset = AutoTrainDataset(
                train_data=[train_data_path],
                task="lm_training",
                token=self.params.token,
                project_name=self.params.project_name,
                username=self.username,
                column_mapping={"text": self.params.text_column},
                valid_data=valid_data_path,
                percent_valid=None,  # TODO: add to UI
            )
            dset.prepare()
            return f"{self.username}/autotrain-data-{self.params.project_name}"

        return self.params.data_path
