import json
from dataclasses import dataclass
from typing import Union

from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.generic.params import GenericParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.object_detection.params import ObjectDetectionParams
from autotrain.trainers.sent_transformers.params import SentenceTransformersParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.text_regression.params import TextRegressionParams
from autotrain.trainers.token_classification.params import TokenClassificationParams


AVAILABLE_HARDWARE = {
    # hugging face spaces
    "spaces-a10g-large": "a10g-large",
    "spaces-a10g-small": "a10g-small",
    "spaces-a100-large": "a100-large",
    "spaces-t4-medium": "t4-medium",
    "spaces-t4-small": "t4-small",
    "spaces-cpu-upgrade": "cpu-upgrade",
    "spaces-cpu-basic": "cpu-basic",
    "spaces-l4x1": "l4x1",
    "spaces-l4x4": "l4x4",
    "spaces-a10g-largex2": "a10g-largex2",
    "spaces-a10g-largex4": "a10g-largex4",
    # ngc
    "dgx-a100": "dgxa100.80g.1.norm",
    "dgx-2a100": "dgxa100.80g.2.norm",
    "dgx-4a100": "dgxa100.80g.4.norm",
    "dgx-8a100": "dgxa100.80g.8.norm",
    # hugging face endpoints
    "ep-aws-useast1-s": "aws_us-east-1_gpu_small_g4dn.xlarge",
    "ep-aws-useast1-m": "aws_us-east-1_gpu_medium_g5.2xlarge",
    "ep-aws-useast1-l": "aws_us-east-1_gpu_large_g4dn.12xlarge",
    "ep-aws-useast1-xl": "aws_us-east-1_gpu_xlarge_p4de",
    "ep-aws-useast1-2xl": "aws_us-east-1_gpu_2xlarge_p4de",
    "ep-aws-useast1-4xl": "aws_us-east-1_gpu_4xlarge_p4de",
    "ep-aws-useast1-8xl": "aws_us-east-1_gpu_8xlarge_p4de",
    # nvcf
    "nvcf-l40sx1": {"id": "67bb8939-c932-429a-a446-8ae898311856"},
    "nvcf-h100x1": {"id": "848348f8-a4e2-4242-bce9-6baa1bd70a66"},
    "nvcf-h100x2": {"id": "fb006a89-451e-4d9c-82b5-33eff257e0bf"},
    "nvcf-h100x4": {"id": "21bae5af-87e5-4132-8fc0-bf3084e59a57"},
    "nvcf-h100x8": {"id": "6e0c2af6-5368-47e0-b15e-c070c2c92018"},
    # local
    "local-ui": "local",
    "local": "local",
    "local-cli": "local",
}


@dataclass
class BaseBackend:
    params: Union[
        TextClassificationParams,
        ImageClassificationParams,
        LLMTrainingParams,
        GenericParams,
        TabularParams,
        DreamBoothTrainingParams,
        Seq2SeqParams,
        TokenClassificationParams,
        TextRegressionParams,
        ObjectDetectionParams,
        SentenceTransformersParams,
    ]
    backend: str

    def __post_init__(self):
        self.username = None

        if isinstance(self.params, GenericParams) and self.backend.startswith("local"):
            raise ValueError("Local backend is not supported for GenericParams")

        if (
            self.backend.startswith("spaces-")
            or self.backend.startswith("ep-")
            or self.backend.startswith("ngc-")
            or self.backend.startswith("nvcf-")
        ):
            if self.params.username is not None:
                self.username = self.params.username
            else:
                raise ValueError("Must provide username")

        if isinstance(self.params, LLMTrainingParams):
            self.task_id = 9
        elif isinstance(self.params, TextClassificationParams):
            self.task_id = 2
        elif isinstance(self.params, TabularParams):
            self.task_id = 26
        elif isinstance(self.params, GenericParams):
            self.task_id = 27
        elif isinstance(self.params, DreamBoothTrainingParams):
            self.task_id = 25
        elif isinstance(self.params, Seq2SeqParams):
            self.task_id = 28
        elif isinstance(self.params, ImageClassificationParams):
            self.task_id = 18
        elif isinstance(self.params, TokenClassificationParams):
            self.task_id = 4
        elif isinstance(self.params, TextRegressionParams):
            self.task_id = 10
        elif isinstance(self.params, ObjectDetectionParams):
            self.task_id = 29
        elif isinstance(self.params, SentenceTransformersParams):
            self.task_id = 30
        else:
            raise NotImplementedError

        self.available_hardware = AVAILABLE_HARDWARE

        self.wait = False
        if self.backend == "local-ui":
            self.wait = False
        if self.backend in ("local", "local-cli"):
            self.wait = True

        self.env_vars = {
            "HF_TOKEN": self.params.token,
            "AUTOTRAIN_USERNAME": self.username,
            "PROJECT_NAME": self.params.project_name,
            "TASK_ID": str(self.task_id),
            "PARAMS": json.dumps(self.params.model_dump_json()),
        }
        if isinstance(self.params, DreamBoothTrainingParams):
            self.env_vars["DATA_PATH"] = self.params.image_path
        else:
            self.env_vars["DATA_PATH"] = self.params.data_path

        if not isinstance(self.params, GenericParams):
            self.env_vars["MODEL"] = self.params.model
