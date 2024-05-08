"""
Copyright 2023 The HuggingFace Team
"""

from dataclasses import dataclass
from typing import List, Union

from autotrain.backends.base import AVAILABLE_HARDWARE
from autotrain.backends.endpoints import EndpointsRunner
from autotrain.backends.local import LocalRunner
from autotrain.backends.ngc import NGCRunner
from autotrain.backends.nvcf import NVCFRunner
from autotrain.backends.spaces import SpaceRunner
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.text_regression.params import TextRegressionParams


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
                TextRegressionParams,
            ]
        ],
        LLMTrainingParams,
        TextClassificationParams,
        TabularParams,
        DreamBoothTrainingParams,
        Seq2SeqParams,
        ImageClassificationParams,
        TextRegressionParams,
    ]
    backend: str

    def __post_init__(self):
        if self.backend not in AVAILABLE_HARDWARE:
            raise ValueError(f"Invalid backend: {self.backend}")

    def create(self):
        if self.backend.startswith("local"):
            runner = LocalRunner(params=self.params, backend=self.backend)
            return runner.create()
        elif self.backend.startswith("spaces-"):
            runner = SpaceRunner(params=self.params, backend=self.backend)
            return runner.create()
        elif self.backend.startswith("ep-"):
            runner = EndpointsRunner(params=self.params, backend=self.backend)
            return runner.create()
        elif self.backend.startswith("ngc-"):
            runner = NGCRunner(params=self.params, backend=self.backend)
            return runner.create()
        elif self.backend.startswith("nvcf-"):
            runner = NVCFRunner(params=self.params, backend=self.backend)
            return runner.create()
        else:
            raise NotImplementedError
