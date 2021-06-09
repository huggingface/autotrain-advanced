from dataclasses import dataclass
from datetime import datetime

from datasets.load import import_main_class, prepare_module

from .utils import BOLD_TAG, GREEN_TAG, RESET_TAG, YELLOW_TAG, get_task


@dataclass
class Evaluate:
    _token: str
    user: str
    task: str
    evaluation_id: int
    created_at: datetime
    updated_at: datetime
    dataset: str
    model: str

    @classmethod
    def from_json_resp(cls, json_resp: dict, token: str):
        task = get_task(json_resp["task"])
        return cls(
            _token=token,
            user=json_resp["username"],
            task=task,
            evaluation_id=json_resp["id"],
            created_at=datetime.fromisoformat(json_resp["created_at"]),
            updated_at=datetime.fromisoformat(json_resp["updated_at"]),
            dataset=json_resp["dataset"],
            model=json_resp["model"],
        )

    def __str__(self):
        output = "\n".join(
            [
                f"AutoNLP Evaluation (id # {self.evaluation_id})",
                "~" * 35,
                f" • {BOLD_TAG}Owner{RESET_TAG}:       {GREEN_TAG}{self.user}{RESET_TAG}",
                f" • {BOLD_TAG}Task{RESET_TAG}:        {YELLOW_TAG}{self.task.title().replace('_', ' ')}{RESET_TAG}",
                f" • {BOLD_TAG}Dataset{RESET_TAG}:       {GREEN_TAG}{self.dataset}{RESET_TAG}",
                f" • {BOLD_TAG}Model{RESET_TAG}:       {GREEN_TAG}{self.model}{RESET_TAG}",
                f" • {BOLD_TAG}Created at{RESET_TAG}:  {self.created_at.strftime('%Y-%m-%d %H:%M Z')}",
                f" • {BOLD_TAG}Last update{RESET_TAG}: {self.updated_at.strftime('%Y-%m-%d %H:%M Z')}",
            ]
        )
        return output


def format_task(task: str, dataset: str, config: str = None):
    task_templates = get_task_templates(dataset, config)
    if task == "text_classification":
        compatible_templates = [template for template in task_templates if template.task == "text-classification"]
        if not compatible_templates:
            raise ValueError(f"Task {task} is not compatible with this dataset! Available tasks: {task_templates}")
        if len(compatible_templates) > 1:
            raise ValueError(
                f"Expected 1 task template but found {len(compatible_templates)}! Please ensure that `datasets.DatasetInfo.task_templates` contains a unique set of task types."
            )
        task_template = compatible_templates[0]
        num_labels = len(task_template.labels)
        if num_labels == 2:
            task = "binary_classification"
        elif num_labels > 2:
            task = "multi_class_classification"
        else:
            raise ValueError(
                f"Dataset `{dataset}` with configuration `{config}` is not suitable for text_classification!"
            )
    return task


def get_task_templates(dataset: str, config: str = None):
    module, module_hash = prepare_module(dataset)
    builder_cls = import_main_class(module)
    builder = builder_cls(hash=module_hash, name=config)
    return builder.info.task_templates
