from autotrain.backends.base import BaseBackend


AVAILABLE_HARDWARE = {
    "spaces-a10g-large": "a10g-large",
    "spaces-a10g-small": "a10g-small",
    "spaces-a100-large": "a100-large",
    "spaces-t4-medium": "t4-medium",
    "spaces-t4-small": "t4-small",
    "spaces-cpu": "cpu-upgrade",
    "spaces-cpu-basic": "cpu-basic",
    "spaces-l4x1": "l4x1",
    "spaces-l4x4": "l4x4",
    "spaces-a10g-largex2": "a10g-largex2",
    "spaces-a10g-largex4": "a10g-largex4",
}


class SpaceRunner(BaseBackend):
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
            space_hardware=self.spaces_backends[self.backend.split("-")[1].lower()],
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
