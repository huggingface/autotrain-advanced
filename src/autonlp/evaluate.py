from dataclasses import dataclass
from datetime import datetime

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
                f" • {BOLD_TAG}Status{RESET_TAG}:      {BOLD_TAG}{self.status_emoji} {self.status}{RESET_TAG}",
                f" • {BOLD_TAG}Task{RESET_TAG}:        {YELLOW_TAG}{self.task.title().replace('_', ' ')}{RESET_TAG}",
                f" • {BOLD_TAG}Dataset{RESET_TAG}:       {GREEN_TAG}{self.dataset}{RESET_TAG}",
                f" • {BOLD_TAG}Model{RESET_TAG}:       {GREEN_TAG}{self.model}{RESET_TAG}",
                f" • {BOLD_TAG}Created at{RESET_TAG}:  {self.created_at.strftime('%Y-%m-%d %H:%M Z')}",
                f" • {BOLD_TAG}Last update{RESET_TAG}: {self.updated_at.strftime('%Y-%m-%d %H:%M Z')}",
            ]
        )
        return output
