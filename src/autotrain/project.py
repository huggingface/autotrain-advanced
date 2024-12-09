"""
Copyright 2023 The HuggingFace Team
"""

import os
from dataclasses import dataclass
from typing import Union

from autotrain.backends.base import AVAILABLE_HARDWARE
from autotrain.backends.endpoints import EndpointsRunner
from autotrain.backends.local import LocalRunner
from autotrain.backends.ngc import NGCRunner
from autotrain.backends.nvcf import NVCFRunner
from autotrain.backends.spaces import SpaceRunner
from autotrain.dataset import (
    AutoTrainDataset,
    AutoTrainImageClassificationDataset,
    AutoTrainImageRegressionDataset,
    AutoTrainObjectDetectionDataset,
    AutoTrainVLMDataset,
)
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.extractive_question_answering.params import ExtractiveQuestionAnsweringParams
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


def tabular_munge_data(params, local):
    if isinstance(params.target_columns, str):
        col_map_label = [params.target_columns]
    else:
        col_map_label = params.target_columns
    task = params.task
    if task == "classification" and len(col_map_label) > 1:
        task = "tabular_multi_label_classification"
    elif task == "classification" and len(col_map_label) == 1:
        task = "tabular_multi_class_classification"
    elif task == "regression" and len(col_map_label) > 1:
        task = "tabular_multi_column_regression"
    elif task == "regression" and len(col_map_label) == 1:
        task = "tabular_single_column_regression"
    else:
        raise Exception("Please select a valid task.")

    exts = ["csv", "jsonl"]
    ext_to_use = None
    for ext in exts:
        path = f"{params.data_path}/{params.train_split}.{ext}"
        if os.path.exists(path):
            ext_to_use = ext
            break

    train_data_path = f"{params.data_path}/{params.train_split}.{ext_to_use}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.{ext_to_use}"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            task=task,
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping={"id": params.id_column, "label": col_map_label},
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            percent_valid=None,  # TODO: add to UI
            local=local,
            ext=ext_to_use,
        )
        params.data_path = dset.prepare()
        params.valid_split = "validation"
        params.id_column = "autotrain_id"
        if len(col_map_label) == 1:
            params.target_columns = ["autotrain_label"]
        else:
            params.target_columns = [f"autotrain_label_{i}" for i in range(len(col_map_label))]
    return params


def llm_munge_data(params, local):
    exts = ["csv", "jsonl"]
    ext_to_use = None
    for ext in exts:
        path = f"{params.data_path}/{params.train_split}.{ext}"
        if os.path.exists(path):
            ext_to_use = ext
            break

    train_data_path = f"{params.data_path}/{params.train_split}.{ext_to_use}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.{ext_to_use}"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        col_map = {"text": params.text_column}
        if params.rejected_text_column is not None:
            col_map["rejected_text"] = params.rejected_text_column
        if params.prompt_text_column is not None:
            col_map["prompt"] = params.prompt_text_column
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            task="lm_training",
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping=col_map,
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            percent_valid=None,  # TODO: add to UI
            local=local,
            ext=ext_to_use,
        )
        params.data_path = dset.prepare()
        params.valid_split = None
        params.text_column = "autotrain_text"
        params.rejected_text_column = "autotrain_rejected_text"
        params.prompt_text_column = "autotrain_prompt"
    return params


def seq2seq_munge_data(params, local):
    exts = ["csv", "jsonl"]
    ext_to_use = None
    for ext in exts:
        path = f"{params.data_path}/{params.train_split}.{ext}"
        if os.path.exists(path):
            ext_to_use = ext
            break

    train_data_path = f"{params.data_path}/{params.train_split}.{ext_to_use}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.{ext_to_use}"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            task="seq2seq",
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping={"text": params.text_column, "label": params.target_column},
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            percent_valid=None,  # TODO: add to UI
            local=local,
            ext=ext_to_use,
        )
        params.data_path = dset.prepare()
        params.valid_split = "validation"
        params.text_column = "autotrain_text"
        params.target_column = "autotrain_label"
    return params


def text_clf_munge_data(params, local):
    exts = ["csv", "jsonl"]
    ext_to_use = None
    for ext in exts:
        path = f"{params.data_path}/{params.train_split}.{ext}"
        if os.path.exists(path):
            ext_to_use = ext
            break

    train_data_path = f"{params.data_path}/{params.train_split}.{ext_to_use}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.{ext_to_use}"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            task="text_multi_class_classification",
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping={"text": params.text_column, "label": params.target_column},
            percent_valid=None,  # TODO: add to UI
            local=local,
            convert_to_class_label=True,
            ext=ext_to_use,
        )
        params.data_path = dset.prepare()
        params.valid_split = "validation"
        params.text_column = "autotrain_text"
        params.target_column = "autotrain_label"
    return params


def text_reg_munge_data(params, local):
    exts = ["csv", "jsonl"]
    ext_to_use = None
    for ext in exts:
        path = f"{params.data_path}/{params.train_split}.{ext}"
        if os.path.exists(path):
            ext_to_use = ext
            break

    train_data_path = f"{params.data_path}/{params.train_split}.{ext_to_use}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.{ext_to_use}"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            task="text_single_column_regression",
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping={"text": params.text_column, "label": params.target_column},
            percent_valid=None,  # TODO: add to UI
            local=local,
            convert_to_class_label=False,
            ext=ext_to_use,
        )
        params.data_path = dset.prepare()
        params.valid_split = "validation"
        params.text_column = "autotrain_text"
        params.target_column = "autotrain_label"
    return params


def token_clf_munge_data(params, local):
    exts = ["csv", "jsonl"]
    ext_to_use = None
    for ext in exts:
        path = f"{params.data_path}/{params.train_split}.{ext}"
        if os.path.exists(path):
            ext_to_use = ext
            break

    train_data_path = f"{params.data_path}/{params.train_split}.{ext_to_use}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.{ext_to_use}"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            task="text_token_classification",
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping={"text": params.tokens_column, "label": params.tags_column},
            percent_valid=None,  # TODO: add to UI
            local=local,
            convert_to_class_label=True,
            ext=ext_to_use,
        )
        params.data_path = dset.prepare()
        params.valid_split = "validation"
        params.tokens_column = "autotrain_text"
        params.tags_column = "autotrain_label"
    return params


def img_clf_munge_data(params, local):
    train_data_path = f"{params.data_path}/{params.train_split}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}"
    else:
        valid_data_path = None
    if os.path.isdir(train_data_path):
        dset = AutoTrainImageClassificationDataset(
            train_data=train_data_path,
            valid_data=valid_data_path,
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            local=local,
        )
        params.data_path = dset.prepare()
        params.valid_split = "validation"
        params.image_column = "autotrain_image"
        params.target_column = "autotrain_label"
    return params


def img_obj_detect_munge_data(params, local):
    train_data_path = f"{params.data_path}/{params.train_split}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}"
    else:
        valid_data_path = None
    if os.path.isdir(train_data_path):
        dset = AutoTrainObjectDetectionDataset(
            train_data=train_data_path,
            valid_data=valid_data_path,
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            local=local,
        )
        params.data_path = dset.prepare()
        params.valid_split = "validation"
        params.image_column = "autotrain_image"
        params.objects_column = "autotrain_objects"
    return params


def sent_transformers_munge_data(params, local):
    exts = ["csv", "jsonl"]
    ext_to_use = None
    for ext in exts:
        path = f"{params.data_path}/{params.train_split}.{ext}"
        if os.path.exists(path):
            ext_to_use = ext
            break

    train_data_path = f"{params.data_path}/{params.train_split}.{ext_to_use}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.{ext_to_use}"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            task="sentence_transformers",
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping={
                "sentence1": params.sentence1_column,
                "sentence2": params.sentence2_column,
                "sentence3": params.sentence3_column,
                "target": params.target_column,
            },
            percent_valid=None,  # TODO: add to UI
            local=local,
            convert_to_class_label=True if params.trainer == "pair_class" else False,
            ext=ext_to_use,
        )
        params.data_path = dset.prepare()
        params.valid_split = "validation"
        params.sentence1_column = "autotrain_sentence1"
        params.sentence2_column = "autotrain_sentence2"
        params.sentence3_column = "autotrain_sentence3"
        params.target_column = "autotrain_target"
    return params


def img_reg_munge_data(params, local):
    train_data_path = f"{params.data_path}/{params.train_split}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}"
    else:
        valid_data_path = None
    if os.path.isdir(train_data_path):
        dset = AutoTrainImageRegressionDataset(
            train_data=train_data_path,
            valid_data=valid_data_path,
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            local=local,
        )
        params.data_path = dset.prepare()
        params.valid_split = "validation"
        params.image_column = "autotrain_image"
        params.target_column = "autotrain_label"
    return params


def vlm_munge_data(params, local):
    train_data_path = f"{params.data_path}/{params.train_split}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        col_map = {"text": params.text_column}
        if params.prompt_text_column is not None:
            col_map["prompt"] = params.prompt_text_column
        dset = AutoTrainVLMDataset(
            train_data=train_data_path,
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping=col_map,
            valid_data=valid_data_path if valid_data_path is not None else None,
            percent_valid=None,  # TODO: add to UI
            local=local,
        )
        params.data_path = dset.prepare()
        params.text_column = "autotrain_text"
        params.image_column = "autotrain_image"
        params.prompt_text_column = "autotrain_prompt"
    return params


def ext_qa_munge_data(params, local):
    exts = ["csv", "jsonl"]
    ext_to_use = None
    for ext in exts:
        path = f"{params.data_path}/{params.train_split}.{ext}"
        if os.path.exists(path):
            ext_to_use = ext
            break

    train_data_path = f"{params.data_path}/{params.train_split}.{ext_to_use}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.{ext_to_use}"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            task="text_extractive_question_answering",
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping={
                "text": params.text_column,
                "question": params.question_column,
                "answer": params.answer_column,
            },
            percent_valid=None,  # TODO: add to UI
            local=local,
            convert_to_class_label=True,
            ext=ext_to_use,
        )
        params.data_path = dset.prepare()
        params.valid_split = "validation"
        params.text_column = "autotrain_text"
        params.question_column = "autotrain_question"
        params.answer_column = "autotrain_answer"
    return params


@dataclass
class AutoTrainProject:
    """
    A class to train an AutoTrain project

    Attributes
    ----------
    params : Union[
        LLMTrainingParams,
        TextClassificationParams,
        TabularParams,
        Seq2SeqParams,
        ImageClassificationParams,
        TextRegressionParams,
        ObjectDetectionParams,
        TokenClassificationParams,
        SentenceTransformersParams,
        ImageRegressionParams,
        ExtractiveQuestionAnsweringParams,
        VLMTrainingParams,
    ]
        The parameters for the AutoTrain project.
    backend : str
        The backend to be used for the AutoTrain project. It should be one of the following:
        - local
        - spaces-a10g-large
        - spaces-a10g-small
        - spaces-a100-large
        - spaces-t4-medium
        - spaces-t4-small
        - spaces-cpu-upgrade
        - spaces-cpu-basic
        - spaces-l4x1
        - spaces-l4x4
        - spaces-l40sx1
        - spaces-l40sx4
        - spaces-l40sx8
        - spaces-a10g-largex2
        - spaces-a10g-largex4
    process : bool
        Flag to indicate if the params and dataset should be processed. If your data format is not AutoTrain-readable, set it to True. Set it to True when in doubt. Defaults to False.

    Methods
    -------
    __post_init__():
        Validates the backend attribute.
    create():
        Creates a runner based on the backend and initializes the AutoTrain project.
    """

    params: Union[
        LLMTrainingParams,
        TextClassificationParams,
        TabularParams,
        Seq2SeqParams,
        ImageClassificationParams,
        TextRegressionParams,
        ObjectDetectionParams,
        TokenClassificationParams,
        SentenceTransformersParams,
        ImageRegressionParams,
        ExtractiveQuestionAnsweringParams,
        VLMTrainingParams,
    ]
    backend: str
    process: bool = False

    def __post_init__(self):
        self.local = self.backend.startswith("local")
        if self.backend not in AVAILABLE_HARDWARE:
            raise ValueError(f"Invalid backend: {self.backend}")

    def _process_params_data(self):
        if isinstance(self.params, LLMTrainingParams):
            return llm_munge_data(self.params, self.local)
        elif isinstance(self.params, ExtractiveQuestionAnsweringParams):
            return ext_qa_munge_data(self.params, self.local)
        elif isinstance(self.params, ImageClassificationParams):
            return img_clf_munge_data(self.params, self.local)
        elif isinstance(self.params, ImageRegressionParams):
            return img_reg_munge_data(self.params, self.local)
        elif isinstance(self.params, ObjectDetectionParams):
            return img_obj_detect_munge_data(self.params, self.local)
        elif isinstance(self.params, SentenceTransformersParams):
            return sent_transformers_munge_data(self.params, self.local)
        elif isinstance(self.params, Seq2SeqParams):
            return seq2seq_munge_data(self.params, self.local)
        elif isinstance(self.params, TabularParams):
            return tabular_munge_data(self.params, self.local)
        elif isinstance(self.params, TextClassificationParams):
            return text_clf_munge_data(self.params, self.local)
        elif isinstance(self.params, TextRegressionParams):
            return text_reg_munge_data(self.params, self.local)
        elif isinstance(self.params, TokenClassificationParams):
            return token_clf_munge_data(self.params, self.local)
        elif isinstance(self.params, VLMTrainingParams):
            return vlm_munge_data(self.params, self.local)
        else:
            raise Exception("Invalid params class")

    def create(self):
        if self.process:
            self.params = self._process_params_data()

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
