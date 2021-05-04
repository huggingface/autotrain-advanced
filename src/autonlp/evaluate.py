from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

from .tasks import TASKS
from .utils import BOLD_TAG, GREEN_TAG, RESET_TAG, YELLOW_TAG


JOB_STATUS = (
    ("⌚", "queued"),
    ("🚀", "start"),
    ("⚙", "data_munging"),
    ("🏃", "model_evaluating"),
    ("✅", "success"),
    ("❌", "failed"),
)


def get_task(task_id: int) -> str:
    for key, value in TASKS.items():
        if value == task_id:
            return key
    return "❌ Unsupported task! Please update autonlp"


def get_eval_job_status(status_id: int) -> Tuple[str, str]:
    try:
        return JOB_STATUS[status_id - 1]
    except IndexError:
        return "❓", "Unhandled status! Please update autonlp"


@dataclass
class Evaluate:
    _token: str
    user: str
    task: str
    status_emoji: str
    status: str
    evaluation_id: int
    created_at: datetime
    updated_at: datetime
    dataset: str
    model: str

    @classmethod
    def from_json_resp(cls, json_resp: dict, token: str):
        task = get_task(json_resp["task"])
        status_emoji, status = get_eval_job_status(json_resp["status"])
        return cls(
            _token=token,
            user=json_resp["username"],
            task=task,
            status_emoji=status_emoji,
            status=status,
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
                f" • {BOLD_TAG}Status{RESET_TAG}:      {BOLD_TAG}{self.status_emoji} {self.status}{RESET_TAG}",
                f" • {BOLD_TAG}Task{RESET_TAG}:        {YELLOW_TAG}{self.task.title().replace('_', ' ')}{RESET_TAG}",
                f" • {BOLD_TAG}Dataset{RESET_TAG}:       {GREEN_TAG}{self.dataset}{RESET_TAG}",
                f" • {BOLD_TAG}Model{RESET_TAG}:       {GREEN_TAG}{self.model}{RESET_TAG}",
                f" • {BOLD_TAG}Created at{RESET_TAG}:  {self.created_at.strftime('%Y-%m-%d %H:%M Z')}",
                f" • {BOLD_TAG}Last update{RESET_TAG}: {self.updated_at.strftime('%Y-%m-%d %H:%M Z')}",
            ]
        )
        return output
