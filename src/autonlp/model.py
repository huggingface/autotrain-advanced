from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from prettytable import PrettyTable

from .utils import BOLD_TAG, PURPLE_TAG, RESET_TAG


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

        # gp.plot(
        #     (np.arange(len(train_losses)), np.asarray(train_losses)),
        #     _with="lines",
        #     terminal="dumb 50,15",
        #     unset="grid",
        # )

        printout.append("")
        printout.append("~" * 35)
        printout.append("")

        print("\n".join(printout))
        printout = []

        printout.append("⭐️ Validation Log:")
        valid_losses = []
        for log in valid_log:
            print(log)
            printout.append(
                f" • {BOLD_TAG}Epoch:{RESET_TAG} {log['epoch']}, {PURPLE_TAG}Loss: {log['eval_loss']}{RESET_TAG}"
            )
            valid_losses.append(log["eval_loss"])

        print("\n".join(printout))
