from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel

from autotrain.tasks import TASKS


class Recipe(BaseModel):
    task: str
    params: dict
    hardware: str


@dataclass
class Recipes:
    task: str
    quantization: Optional[str] = None

    def __post_init__(self):
        self.task = self.task.lower()
        if self.task not in TASKS:
            raise ValueError(f"Task {self.task} not found in TASKS")

        if self.quantization not in ("int4", "int8", None):
            raise ValueError(f"Quantization {self.quantization} not supported")

        if self.task == "text_binary_classification":
            self.task = "text_multi_class_classification"

    def get_recipe(self):
        if self.task == "text_multi_class_classification":
            return Recipe(
                task=self.task, params={"model": "distilbert-base-uncased", "max_length": 128}, hardware="auto"
            )
