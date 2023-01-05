from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from prettytable import PrettyTable


@dataclass
class Model:
    _token: str
    json_resp: Union[List, List[Dict[str, float]]]
    username: str
    model_id: int
    max_train_print_count: Optional[int] = 7

    @classmethod
    def from_json_resp(cls, json_resp: Union[List, List[Dict[str, float]]], token: str, username: str, model_id: str):
        return cls(_token=token, json_resp=json_resp, username=username, model_id=model_id)

    def print(self):
        printout = ["~" * 35, f"🚀 AutoNLP Model Info for model_id # {self.model_id}", "~" * 35, ""]
        if len(self.json_resp) == 0:
            printout.append("🚨 Model ID found but has no entries yet!!!")
            print("\n".join(printout))

        training_log = [log for log in self.json_resp if "loss" in log]
        valid_log = [log for log in self.json_resp if "eval_loss" in log]

        printout.append("⭐️ Training Log:")
        print_training_logs = PrettyTable(["Epoch", "Loss"])
        # train_losses = [log["loss"] for log in training_log]
        for log in training_log:
            print_training_logs.add_row([log["epoch"], log["loss"]])

        print("\n".join(printout))
        print(print_training_logs)
        printout = []

        printout.append("")
        printout.append("~" * 35)
        printout.append("")

        print("\n".join(printout))
        printout = []

        printout.append("⭐️ Validation Log:")
        print_validation_logs = PrettyTable(["Epoch", "Loss"])
        for log in valid_log:
            print_validation_logs.add_row([log["epoch"], log["eval_loss"]])

        print("\n".join(printout))
        print(print_validation_logs)
