"""
Copyright 2023 The HuggingFace Team
"""

from dataclasses import dataclass
from typing import List, Union

from autotrain.backend import SpaceRunner
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams


@dataclass
class AutoTrainProject:
    params: Union[
        List[
            Union[
                LLMTrainingParams,
                TextClassificationParams,
                TabularParams,
                DreamBoothTrainingParams,
                Seq2SeqParams,
                ImageClassificationParams,
            ]
        ],
        LLMTrainingParams,
        TextClassificationParams,
        TabularParams,
        DreamBoothTrainingParams,
        Seq2SeqParams,
        ImageClassificationParams,
    ]
    backend: str

    def __post_init__(self):
        self.spaces_backends = {
            "A10G Large": "spaces-a10gl",
            "A10G Small": "spaces-a10gs",
            "A100 Large": "spaces-a100",
            "T4 Medium": "spaces-t4m",
            "T4 Small": "spaces-t4s",
            "CPU Upgrade": "spaces-cpu",
            "CPU (Free)": "spaces-cpuf",
            "DGX 1xA100": "dgx-a100",
            "DGX 2xA100": "dgx-2a100",
            "DGX 4xA100": "dgx-4a100",
            "DGX 8xA100": "dgx-8a100",
            "NVCF 1xH100": "nvcf-h100x1",
            "NVCF 1xL40": "nvcf-l40",
            "Local": "local",
            "EP US-East-1 1xA10g": "ep-aws-useast1-m",
            "EP US-East-1 1xA100": "ep-aws-useast1-xl",
            "EP US-East-1 2xA100": "ep-aws-useast1-2xl",
            "EP US-East-1 4xA100": "ep-aws-useast1-4xl",
            "EP US-East-1 8xA100": "ep-aws-useast1-8xl",
            "local": "local",
            "local-cli": "local-cli",
        }

    def create_spaces(self):
        sr = SpaceRunner(params=self.params, backend=self.spaces_backends[self.backend])
        space_id = sr.prepare()
        return space_id

    def create(self):
        if self.backend == "AutoTrain":
            raise NotImplementedError
        if self.backend in self.spaces_backends:
            return self.create_spaces()
