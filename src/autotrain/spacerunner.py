import json

from huggingface_hub import HfApi


DOCKERFILE = """
FROM huggingface/autotrain-advanced:latest

CMD autotrain api --port 7860
"""


def create_space(user_token, autotrain_username, project_name, instance_type, data_path, params, task):
    api = HfApi(token=user_token)
    repo_id = f"{autotrain_username}/{project_name}"
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        space_hardware=instance_type,
        private=True,
    )
    api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value=user_token)
    api.add_space_secret(repo_id=repo_id, key="AUTOTRAIN_USERNAME", value=autotrain_username)
    api.add_space_secret(repo_id=repo_id, key="PROJECT_NAME", value=project_name)
    # convert params dict to string
    params_str = json.dumps(params)
    api.add_space_secret(repo_id=repo_id, key="PARAMS", value=params_str)
    api.add_space_secret(repo_id=repo_id, key="DATA_PATH", value=data_path)
    api.add_space_secret(repo_id=repo_id, key="TASK", value=task)
