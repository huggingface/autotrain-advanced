from dataclasses import dataclass

import requests
from prettytable import PrettyTable

from .utils import BOLD_TAG, RESET_TAG, http_get


@dataclass
class Metrics:
    _token: str
    username: str
    project_name: str
    project_id: int
    language: str

    @classmethod
    def from_json_resp(cls, json_resp: dict, token: str, project_name: str, username: str):
        return cls(
            project_id=json_resp["id"],
            language=json_resp["config"]["language"],
            _token=token,
            project_name=project_name,
            username=username,
        )

    def model_metrics(self, model_id):
        json_resp = http_get(f"/models/{self.username}/{model_id}", token=self._token, suppress_logs=True).json()
        eval_dicts = [ed for ed in json_resp if "eval_loss" in ed]
        eval_dicts = sorted(eval_dicts, key=lambda k: k["eval_loss"])
        if len(eval_dicts) > 0:
            return eval_dicts[0]

    def print(self):
        resp = http_get(path=f"/projects/{self.project_id}/jobs", token=self._token, suppress_logs=True)
        json_jobs = resp.json()
        printout = ["~" * 35, f"🚀 AutoNLP Model Metrics for project: {self.project_name}", "~" * 35, ""]
        printout.append(f"💾 you have {len(json_jobs)} models in this project")
        print("\n".join(printout))
        print("")
        best_loss = 99999999
        best_model = None
        print_logs = PrettyTable([f"📚 {BOLD_TAG}Model ID{RESET_TAG}", f"📊 {BOLD_TAG}Eval Loss{RESET_TAG}"])
        for job in json_jobs:
            model_id = job["id"]
            try:
                metrics = self.model_metrics(model_id=model_id)
                eval_loss = metrics["eval_loss"]
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    best_model = model_id
                print_logs.add_row([model_id, eval_loss])
            except (requests.exceptions.HTTPError, TypeError):
                continue
        print(print_logs)
        print("")
        print(f"🎖 Best Model: {best_model}, Best Loss: {best_loss}")
