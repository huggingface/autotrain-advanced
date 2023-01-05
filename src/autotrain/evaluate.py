from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

from datasets.load import import_main_class, prepare_module

from .utils import BOLD_TAG, GREEN_TAG, RESET_TAG, YELLOW_TAG, get_task


EVAL_JOB_STATUS = (
    ("⌚", "queued"),
    ("🚀", "start"),
    ("⚙", "data_munging"),
    ("🏃", "model_evaluating"),
    ("✅", "success"),
    ("❌", "failed"),
)


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


def get_eval_job_status(status_id: int) -> Tuple[str, str]:
    try:
        return EVAL_JOB_STATUS[status_id - 1]
    except IndexError:
        return "❓", "Unhandled status! Please update autonlp"


def format_eval_status(json_resp):
    status_emoji, status = get_eval_job_status(json_resp["status"])
    output = "\n".join(
        [
            f"AutoNLP Evaluation (id # {json_resp['id']})",
            "~" * 35,
            f" • {BOLD_TAG}Status{RESET_TAG}:       {status_emoji} {GREEN_TAG}{status}{RESET_TAG}",
        ]
    )
    return output


def format_datasets_task(task: str, dataset: str, config: str = None):
    task_template = get_compatible_task_template(task, dataset, config)
    if task_template:
        if task == "text-classification":
            num_labels = len(task_template.labels)
            if num_labels == 2:
                task = "binary_classification"
            # TODO(lewtun): add logic for multilabel classification when implemented in `datasets`
            elif num_labels > 2:
                task = "multi_class_classification"
            else:
                raise Exception("Invalid `num_labels`")
        elif task == "question-answering-extractive":
            task = "extractive_question_answering"
        else:
            raise ValueError(f"❌ Dataset `{dataset}` with configuration `{config}` is not suitable for `{task}`!")
    else:
        raise ValueError(
            f"❌ Dataset `{dataset}` with configuration `{config}` does not have a task template for `{task}`! Please select a different task and/or dataset."
        )
    return task


def get_compatible_task_template(task: str, dataset: str, config: str = None):
    module, module_hash = prepare_module(dataset)
    builder_cls = import_main_class(module)
    builder = builder_cls(hash=module_hash, name=config)
    templates = builder.info.task_templates
    if templates:
        compatible_templates = [template for template in templates if template.task == task]
        if not compatible_templates:
            raise ValueError(f"❌ Task `{task}` is not compatible with dataset `{dataset}`!")
        if len(compatible_templates) > 1:
            raise ValueError(
                f"❌ Expected 1 task template but found {len(compatible_templates)}! Please ensure that `datasets.DatasetInfo.task_templates` contains a unique set of task types."
            )
        return compatible_templates[0]
    else:
        return None


def get_dataset_splits(dataset: str, config: str = None):
    module, module_hash = prepare_module(dataset)
    builder_cls = import_main_class(module)
    builder = builder_cls(hash=module_hash, name=config)
    splits = builder.info.splits.keys()
    return list(splits)
