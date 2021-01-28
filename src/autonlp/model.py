import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from tqdm import tqdm

from .splits import TEST_SPLIT, TRAIN_SPLIT, VALID_SPLIT
from .tasks import TASKS
from .utils import (
    BOLD_TAG,
    CYAN_TAG,
    GREEN_TAG,
    PURPLE_TAG,
    RESET_TAG,
    YELLOW_TAG,
    http_get,
    http_post,
    http_upload_files,
)

import gnuplotlib as gp
import numpy as np


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
        train_print_counter = 0
        train_losses = [log["loss"] for log in training_log]
        for log in training_log[-5:]:
            if train_print_counter < self.max_train_print_count:
                printout.append(
                    f" • {BOLD_TAG}Epoch:{RESET_TAG} {log['epoch']}, {PURPLE_TAG}Loss: {log['loss']}{RESET_TAG}"
                )
                train_print_counter += 1

        print("\n".join(printout))
        printout = []

        gp.plot(
            (np.arange(len(train_losses)), np.asarray(train_losses)),
            _with="lines",
            terminal="dumb 50,15",
            unset="grid",
        )

        printout.append("")
        printout.append("~" * 35)
        printout.append("")

        print("\n".join(printout))
        printout = []

        printout.append("⭐️ Validation Log:")
        valid_losses = []
        for log in valid_log:
            printout.append(
                f" • {BOLD_TAG}Epoch:{RESET_TAG} {log['epoch']}, {PURPLE_TAG}Loss: {log['eval_loss']}{RESET_TAG}"
            )
            valid_losses.append(log["eval_loss"])

        print("\n".join(printout))
