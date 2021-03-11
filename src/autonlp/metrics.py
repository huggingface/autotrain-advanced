import json
from dataclasses import dataclass

from prettytable import PrettyTable

from .utils import BOLD_TAG, RESET_TAG, http_get, http_post


@dataclass
class Metrics:
    _token: str
    username: str
    project_name: str
    project_id: int

    @classmethod
    def from_json_resp(cls, json_resp: dict, token: str, project_name: str, username: str):
        return cls(
            project_id=json_resp["id"],
            _token=token,
            project_name=project_name,
            username=username,
        )

    def print(self):
        payload = {"project_id": self.project_id, "username": self.username}
        resp = http_post(path="/models/metrics", token=self._token, suppress_logs=True, payload=payload)
        json_jobs = json.loads(resp.json())
        if len(json_jobs) > 0:
            columns = json_jobs.keys()
            print_logs = PrettyTable([col for col in columns])
            print_logs.title = f"{BOLD_TAG}🚀 AutoNLP Model Metrics for project: {self.project_name}{RESET_TAG}"
            indexes = json_jobs["model_id"].keys()
            for idx in indexes:
                line = [v[idx] if k == "model_id" else float(v[idx]) for k, v in json_jobs.items()]
                print_logs.add_row(line)

            print_logs = print_logs.get_string(sortby="eval_loss")
            print(print_logs)
            print("")
            resp = http_get(path=f"/zeus/cost/{self.project_id}", token=self._token, suppress_logs=True).json()
            print(f"Total cost incurred: USD {resp.get('cost_usd'):.2f}")
        else:
            print("No models have finished training yet!")
