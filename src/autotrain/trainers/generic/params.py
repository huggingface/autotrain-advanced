from typing import Dict

from pydantic import Field

from autotrain.trainers.common import AutoTrainParams


class GenericParams(AutoTrainParams):
    username: str = Field(None, title="Hugging Face Username")
    project_name: str = Field(None, title="Output directory")
    data_path: str = Field(None, title="Data path")
    token: str = Field(None, title="Hub Token")
    script_path: str = Field(None, title="Script path")
    env: Dict[str, str] = Field(None, title="Environment Variables")
