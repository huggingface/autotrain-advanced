# flake8: noqa
# coding=utf-8
# Copyright 2020 The HuggingFace AutoNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
# pylint: enable=line-too-long

import json
import requests
from typing import Union, List

from . import config


class AutoNLP:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        if not self._verify_key():
            raise Exception("Unable to verify account or no credits left")

    def _verify_key(self):
        resp = requests.post(
            url=f"{config.HF_AUTONLP_BACKEND_API}/verify_api_key",
            data=json.dumps({"key": self.api_key}),
        )
        resp = resp.json()
        if resp["success"] is True and resp["credits_left"] > 0:
            return True
        return False

    def train(self, dataset: Union[str, List[str]]) -> str:
        if isinstance(dataset, str):
            # do something
            # need to push data to endpoint here.
            print("string")
        elif isinstance(dataset, list):
            print("list of strings")

    def deploy(self):
        pass


if __name__ == "__main__":
    anlp = AutoNLP(api_key="fake_api_key")
    anlp.train(dataset="this")
    anlp.train(dataset=["this", "that"])