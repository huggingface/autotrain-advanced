# import json
# from dataclasses import dataclass
# from typing import Union

# from autotrain.trainers.clm.params import LLMTrainingParams
# from autotrain.trainers.extractive_question_answering.params import ExtractiveQuestionAnsweringParams
# from autotrain.trainers.generic.params import GenericParams
# from autotrain.trainers.image_classification.params import ImageClassificationParams
# from autotrain.trainers.image_regression.params import ImageRegressionParams
# from autotrain.trainers.object_detection.params import ObjectDetectionParams
# from autotrain.trainers.sent_transformers.params import SentenceTransformersParams
# from autotrain.trainers.seq2seq.params import Seq2SeqParams
# from autotrain.trainers.tabular.params import TabularParams
# from autotrain.trainers.text_classification.params import TextClassificationParams
# from autotrain.trainers.text_regression.params import TextRegressionParams
# from autotrain.trainers.token_classification.params import TokenClassificationParams
# from autotrain.trainers.vlm.params import VLMTrainingParams
# from autotrain.trainers.automatic_speech_recognition.params import AutomaticSpeechRecognitionParams


# AVAILABLE_HARDWARE = {
#     # hugging face spaces
#     "spaces-a10g-large": "a10g-large",
#     "spaces-a10g-small": "a10g-small",
#     "spaces-a100-large": "a100-large",
#     "spaces-t4-medium": "t4-medium",
#     "spaces-t4-small": "t4-small",
#     "spaces-cpu-upgrade": "cpu-upgrade",
#     "spaces-cpu-basic": "cpu-basic",
#     "spaces-l4x1": "l4x1",
#     "spaces-l4x4": "l4x4",
#     "spaces-l40sx1": "l40sx1",
#     "spaces-l40sx4": "l40sx4",
#     "spaces-l40sx8": "l40sx8",
#     "spaces-a10g-largex2": "a10g-largex2",
#     "spaces-a10g-largex4": "a10g-largex4",
#     # ngc
#     "dgx-a100": "dgxa100.80g.1.norm",
#     "dgx-2a100": "dgxa100.80g.2.norm",
#     "dgx-4a100": "dgxa100.80g.4.norm",
#     "dgx-8a100": "dgxa100.80g.8.norm",
#     # hugging face endpoints
#     "ep-aws-useast1-s": "aws_us-east-1_gpu_small_g4dn.xlarge",
#     "ep-aws-useast1-m": "aws_us-east-1_gpu_medium_g5.2xlarge",
#     "ep-aws-useast1-l": "aws_us-east-1_gpu_large_g4dn.12xlarge",
#     "ep-aws-useast1-xl": "aws_us-east-1_gpu_xlarge_p4de",
#     "ep-aws-useast1-2xl": "aws_us-east-1_gpu_2xlarge_p4de",
#     "ep-aws-useast1-4xl": "aws_us-east-1_gpu_4xlarge_p4de",
#     "ep-aws-useast1-8xl": "aws_us-east-1_gpu_8xlarge_p4de",
#     # nvcf
#     "nvcf-l40sx1": {"id": "67bb8939-c932-429a-a446-8ae898311856"},
#     "nvcf-h100x1": {"id": "848348f8-a4e2-4242-bce9-6baa1bd70a66"},
#     "nvcf-h100x2": {"id": "fb006a89-451e-4d9c-82b5-33eff257e0bf"},
#     "nvcf-h100x4": {"id": "21bae5af-87e5-4132-8fc0-bf3084e59a57"},
#     "nvcf-h100x8": {"id": "6e0c2af6-5368-47e0-b15e-c070c2c92018"},
#     # local
#     "local-ui": "local",
#     "local": "local",
#     "local-cli": "local",
# }


# @dataclass
# class BaseBackend:
#     """
#     BaseBackend class is responsible for initializing and validating backend configurations
#     for various training parameters. It supports multiple types of training parameters
#     including text classification, image classification, LLM training, and more.

#     Attributes:
#         params (Union[TextClassificationParams, ImageClassificationParams, LLMTrainingParams,
#                       GenericParams, TabularParams, Seq2SeqParams,
#                       TokenClassificationParams, TextRegressionParams, ObjectDetectionParams,
#                       SentenceTransformersParams, ImageRegressionParams, VLMTrainingParams,
#                       ExtractiveQuestionAnsweringParams,AutomaticSpeechRecognitionParams]): Training parameters.
#         backend (str): Backend type.

#     Methods:
#         __post_init__(): Initializes the backend configuration, validates parameters,
#                          sets task IDs, and prepares environment variables.
#     """

#     params: Union[
#         TextClassificationParams,
#         ImageClassificationParams,
#         LLMTrainingParams,
#         GenericParams,
#         TabularParams,
#         Seq2SeqParams,
#         TokenClassificationParams,
#         TextRegressionParams,
#         ObjectDetectionParams,
#         SentenceTransformersParams,
#         ImageRegressionParams,
#         VLMTrainingParams,
#         ExtractiveQuestionAnsweringParams,
#         AutomaticSpeechRecognitionParams,
#     ]
#     backend: str

#     def __post_init__(self):
#         self.username = None

#         if isinstance(self.params, GenericParams) and self.backend.startswith("local"):
#             raise ValueError("Local backend is not supported for GenericParams")

#         if (
#             self.backend.startswith("spaces-")
#             or self.backend.startswith("ep-")
#             or self.backend.startswith("ngc-")
#             or self.backend.startswith("nvcf-")
#         ):
#             if self.params.username is not None:
#                 self.username = self.params.username
#             else:
#                 raise ValueError("Must provide username")

#         if isinstance(self.params, LLMTrainingParams):
#             self.task_id = 9
#         elif isinstance(self.params, TextClassificationParams):
#             self.task_id = 2
#         elif isinstance(self.params, TabularParams):
#             self.task_id = 26
#         elif isinstance(self.params, GenericParams):
#             self.task_id = 27
#         elif isinstance(self.params, Seq2SeqParams):
#             self.task_id = 28
#         elif isinstance(self.params, ImageClassificationParams):
#             self.task_id = 18
#         elif isinstance(self.params, TokenClassificationParams):
#             self.task_id = 4
#         elif isinstance(self.params, TextRegressionParams):
#             self.task_id = 10
#         elif isinstance(self.params, ObjectDetectionParams):
#             self.task_id = 29
#         elif isinstance(self.params, SentenceTransformersParams):
#             self.task_id = 30
#         elif isinstance(self.params, ImageRegressionParams):
#             self.task_id = 24
#         elif isinstance(self.params, VLMTrainingParams):
#             self.task_id = 31
#         elif isinstance(self.params, ExtractiveQuestionAnsweringParams):
#             self.task_id = 5
#         elif isinstance(self.params, AutomaticSpeechRecognitionParams):
#             self.task_id = 32
#         else:
#             raise NotImplementedError

#         self.available_hardware = AVAILABLE_HARDWARE

#         self.wait = False
#         if self.backend == "local-ui":
#             self.wait = False
#         if self.backend in ("local", "local-cli"):
#             self.wait = True

#         self.env_vars = {
#             "HF_TOKEN": self.params.token,
#             "AUTOTRAIN_USERNAME": self.username,
#             "PROJECT_NAME": self.params.project_name,
#             "TASK_ID": str(self.task_id),
#             "PARAMS": json.dumps(self.params.model_dump_json()),
#         }
#         self.env_vars["DATA_PATH"] = self.params.data_path

#         if not isinstance(self.params, GenericParams):
#             self.env_vars["MODEL"] = self.params.model







import json
from dataclasses import dataclass
from typing import Union

from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.extractive_question_answering.params import ExtractiveQuestionAnsweringParams
from autotrain.trainers.generic.params import GenericParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.image_regression.params import ImageRegressionParams
from autotrain.trainers.object_detection.params import ObjectDetectionParams
from autotrain.trainers.sent_transformers.params import SentenceTransformersParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.text_regression.params import TextRegressionParams
from autotrain.trainers.token_classification.params import TokenClassificationParams
from autotrain.trainers.vlm.params import VLMTrainingParams
from autotrain.trainers.automatic_speech_recognition.params import AutomaticSpeechRecognitionParams


AVAILABLE_HARDWARE = {
   
    "spaces-a10g-large": "a10g-large",
    "spaces-a10g-small": "a10g-small",
    "spaces-a100-large": "a100-large",
    "spaces-t4-medium": "t4-medium",
    "spaces-t4-small": "t4-small",
    "spaces-cpu-upgrade": "cpu-upgrade",
    "spaces-cpu-basic": "cpu-basic",
    "spaces-l4x1": "l4x1",
    "spaces-l4x4": "l4x4",
    "spaces-l40sx1": "l40sx1",
    "spaces-l40sx4": "l40sx4",
    "spaces-l40sx8": "l40sx8",
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
    """
    BaseBackend class is responsible for initializing and validating backend configurations
    for various training parameters. It supports multiple types of training parameters
    including text classification, image classification, LLM training, and more.

    Attributes:
        params (Union[TextClassificationParams, ImageClassificationParams, LLMTrainingParams,
                      GenericParams, TabularParams, Seq2SeqParams,
                      TokenClassificationParams, TextRegressionParams, ObjectDetectionParams,
                      SentenceTransformersParams, ImageRegressionParams, VLMTrainingParams,
                      ExtractiveQuestionAnsweringParams, AutomaticSpeechRecognitionParams]): Training parameters.
        backend (str): Backend type.

    Methods:
        __post_init__(): Initializes the backend configuration, validates parameters,
                         sets task IDs, and prepares environment variables.
    """

    params: Union[
        TextClassificationParams,
        ImageClassificationParams,
        LLMTrainingParams,
        GenericParams,
        TabularParams,
        Seq2SeqParams,
        TokenClassificationParams,
        TextRegressionParams,
        ObjectDetectionParams,
        SentenceTransformersParams,
        ImageRegressionParams,
        VLMTrainingParams,
        ExtractiveQuestionAnsweringParams,
        AutomaticSpeechRecognitionParams,
    ]
    backend: str

    def __post_init__(self):
        """Initialize environment variables and task ID after object creation."""
        self.username = None

        if isinstance(self.params, dict):
            task_id = self.params.get("task_id", 32)
            self.task_id = task_id
            base_params = {
                "project_name": self.params.get("project_name", "autotrain-project"),
                "token": self.params.get("token", ""),
                "username": self.params.get("username", ""),
                "data_path": str(self.params.get("data_path", "")),
                "model": self.params.get("model", "facebook/wav2vec2-base"),
                "task_id": task_id,
                "max_duration": self.params.get("max_duration", 30.0),
                "sampling_rate": self.params.get("sampling_rate", 16000),
                "audio_column": self.params.get("audio_column", "audio"),
                "text_column": self.params.get("text_column", "text"),
                "max_grad_norm": self.params.get("max_grad_norm", 1.0),
                "weight_decay": self.params.get("weight_decay", 0.01),
                "warmup_ratio": self.params.get("warmup_ratio", 0.1),
                "early_stopping_patience": self.params.get("early_stopping_patience", 3),
                "early_stopping_threshold": self.params.get("early_stopping_threshold", 0.01),
                "eval_strategy": self.params.get("eval_strategy", "epoch"),
                "save_total_limit": self.params.get("save_total_limit", 1),
                "auto_find_batch_size": self.params.get("auto_find_batch_size", False),
                "logging_steps": self.params.get("logging_steps", -1)
            }
            self.params = AutomaticSpeechRecognitionParams(**base_params)
        elif isinstance(self.params, str):
            params_dict = json.loads(self.params)
            task_id = params_dict.get("task_id", 32)
            self.task_id = task_id
            base_params = {
                "project_name": params_dict.get("project_name", "autotrain-project"),
                "token": params_dict.get("token", ""),
                "username": params_dict.get("username", ""),
                "data_path": str(params_dict.get("data_path", "")),
                "model": params_dict.get("model", "facebook/wav2vec2-base"),
                "task_id": task_id,
                "max_duration": params_dict.get("max_duration", 30.0),
                "sampling_rate": params_dict.get("sampling_rate", 16000),
                "audio_column": params_dict.get("audio_column", "audio"),
                "text_column": params_dict.get("text_column", "text"),
                "max_grad_norm": params_dict.get("max_grad_norm", 1.0),
                "weight_decay": params_dict.get("weight_decay", 0.01),
                "warmup_ratio": params_dict.get("warmup_ratio", 0.1),
                "early_stopping_patience": params_dict.get("early_stopping_patience", 3),
                "early_stopping_threshold": params_dict.get("early_stopping_threshold", 0.01),
                "eval_strategy": params_dict.get("eval_strategy", "epoch"),
                "save_total_limit": params_dict.get("save_total_limit", 1),
                "auto_find_batch_size": params_dict.get("auto_find_batch_size", False),
                "logging_steps": params_dict.get("logging_steps", -1)
            }
            self.params = AutomaticSpeechRecognitionParams(**base_params)
        else:
            if isinstance(self.params, LLMTrainingParams):
                self.task_id = 9
            elif isinstance(self.params, TextClassificationParams):
                self.task_id = 2
            elif isinstance(self.params, TabularParams):
                self.task_id = 26
            elif isinstance(self.params, GenericParams):
                self.task_id = 27
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
            elif isinstance(self.params, ImageRegressionParams):
                self.task_id = 24
            elif isinstance(self.params, VLMTrainingParams):
                self.task_id = 31
            elif isinstance(self.params, ExtractiveQuestionAnsweringParams):
                self.task_id = 5
            elif isinstance(self.params, AutomaticSpeechRecognitionParams):
                self.task_id = 32
                if not hasattr(self.params, 'data_path') or self.params.data_path is None:
                    self.params.data_path = ""
                else:
                    self.params.data_path = str(self.params.data_path)
            else:
                raise NotImplementedError(f"Unknown parameter type: {type(self.params)}")

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
                raise ValueError("Must provide username for non-local backends")
        else:
            
            self.username = self.params.username if self.params.username is not None else ""

        self.available_hardware = AVAILABLE_HARDWARE

        self.wait = False
        if self.backend == "local-ui":
            self.wait = False
        if self.backend in ("local", "local-cli"):
            self.wait = True

        self.env_vars = {
            "PROJECT_NAME": self.params.project_name,
            "TASK_ID": str(self.task_id),
            "PARAMS": json.dumps(self.params.__dict__),
            "HF_TOKEN": self.params.token,
            "HF_USERNAME": self.params.username,
            "BACKEND": self.backend,
            "DATA_PATH": str(self.params.data_path),
        }

        if not isinstance(self.params, GenericParams):
            self.env_vars["MODEL"] = self.params.model

    def _get_task_specific_requirements(self):
        """Get task-specific requirements."""
        if self.task == "ASR":
            return [
                "librosa>=0.10.0",
                "soundfile>=0.12.1",
            ]
        
