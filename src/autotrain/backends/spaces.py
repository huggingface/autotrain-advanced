import io

from huggingface_hub import HfApi

from autotrain.backends.base import BaseBackend
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.generic.params import GenericParams


_DOCKERFILE = """
FROM huggingface/autotrain-advanced:latest

CMD pip uninstall -y autotrain-advanced && pip install -U autotrain-advanced && autotrain api --port 7860 --host 0.0.0.0
"""

# format _DOCKERFILE
_DOCKERFILE = _DOCKERFILE.replace("\n", " ").replace("  ", "\n").strip()


class SpaceRunner(BaseBackend):
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

    def _add_secrets(self, api, space_id):
        if isinstance(self.params, GenericParams):
            for k, v in self.params.env.items():
                api.add_space_secret(repo_id=space_id, key=k, value=v)
            self.params.env = {}

        api.add_space_secret(repo_id=space_id, key="HF_TOKEN", value=self.params.token)
        api.add_space_secret(repo_id=space_id, key="AUTOTRAIN_USERNAME", value=self.username)
        api.add_space_secret(repo_id=space_id, key="PROJECT_NAME", value=self.params.project_name)
        api.add_space_secret(repo_id=space_id, key="TASK_ID", value=str(self.task_id))
        api.add_space_secret(repo_id=space_id, key="PARAMS", value=self.params.model_dump_json())

        if isinstance(self.params, DreamBoothTrainingParams):
            api.add_space_secret(repo_id=space_id, key="DATA_PATH", value=self.params.image_path)
        else:
            api.add_space_secret(repo_id=space_id, key="DATA_PATH", value=self.params.data_path)

        if not isinstance(self.params, GenericParams):
            api.add_space_secret(repo_id=space_id, key="MODEL", value=self.params.model)

    def create(self):
        api = HfApi(token=self.params.token)
        space_id = f"{self.username}/autotrain-{self.params.project_name}"
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="docker",
            space_hardware=self.available_hardware[self.backend],
            private=True,
        )
        self._add_secrets(api, space_id)
        readme = self._create_readme()
        api.upload_file(
            path_or_fileobj=readme,
            path_in_repo="README.md",
            repo_id=space_id,
            repo_type="space",
        )

        _dockerfile = io.BytesIO(_DOCKERFILE.encode())
        api.upload_file(
            path_or_fileobj=_dockerfile,
            path_in_repo="Dockerfile",
            repo_id=space_id,
            repo_type="space",
        )
        return space_id
