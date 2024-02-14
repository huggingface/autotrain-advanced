from typing import Dict, Optional

from pydantic import Field

from autotrain.trainers.common import AutoTrainParams


class GenericParams(AutoTrainParams):
    username: str = Field(None, title="Hugging Face Username")
    project_name: str = Field("project-name", title="path to script.py")
    data_path: str = Field(None, title="Data path")
    token: str = Field(None, title="Hub Token")
    script_path: str = Field(None, title="Script path")
    repo_id: Optional[str] = Field(None, title="Repo ID")
    env: Optional[Dict[str, str]] = Field(None, title="Environment Variables")
    args: Optional[Dict[str, str]] = Field(None, title="Arguments")
