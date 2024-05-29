import os
from typing import Any, Type

from autotrain.backends.base import AVAILABLE_HARDWARE
from autotrain.dataset import (
    AutoTrainDataset,
    AutoTrainDreamboothDataset,
    AutoTrainImageClassificationDataset,
    AutoTrainObjectDetectionDataset,
)


def common_args():
    args = [
        {
            "arg": "--train",
            "help": "Command to train the model",
            "required": False,
            "action": "store_true",
        },
        {
            "arg": "--deploy",
            "help": "Command to deploy the model (limited availability)",
            "required": False,
            "action": "store_true",
        },
        {
            "arg": "--inference",
            "help": "Command to run inference (limited availability)",
            "required": False,
            "action": "store_true",
        },
        {
            "arg": "--username",
            "help": "Hugging Face Hub Username",
            "required": False,
            "type": str,
        },
        {
            "arg": "--backend",
            "help": "Backend to use: default or spaces. Spaces backend requires push_to_hub & username. Advanced users only.",
            "required": False,
            "type": str,
            "default": "local",
            "choices": AVAILABLE_HARDWARE.keys(),
        },
        {
            "arg": "--token",
            "help": "Your Hugging Face API token. Token must have write access to the model hub.",
            "required": False,
            "type": str,
        },
        {
            "arg": "--push-to-hub",
            "help": "Push to hub after training will push the trained model to the Hugging Face model hub.",
            "required": False,
            "action": "store_true",
        },
        {
            "arg": "--model",
            "help": "Base model to use for training",
            "required": True,
            "type": str,
        },
        {
            "arg": "--project-name",
            "help": "Output directory / repo id for trained model (must be unique on hub)",
            "required": True,
            "type": str,
        },
        {
            "arg": "--data-path",
            "help": "Train dataset to use. When using cli, this should be a directory path containing training and validation data in appropriate formats",
            "required": False,
            "type": str,
        },
        {
            "arg": "--train-split",
            "help": "Train dataset split to use",
            "required": False,
            "type": str,
            "default": "train",
        },
        {
            "arg": "--valid-split",
            "help": "Validation dataset split to use",
            "required": False,
            "type": str,
            "default": None,
        },
        {
            "arg": "--batch-size",
            "help": "Training batch size to use",
            "required": False,
            "type": int,
            "default": 2,
            "alias": ["--train-batch-size"],
        },
        {
            "arg": "--seed",
            "help": "Random seed for reproducibility",
            "required": False,
            "default": 42,
            "type": int,
        },
        {
            "arg": "--epochs",
            "help": "Number of training epochs",
            "required": False,
            "default": 1,
            "type": int,
        },
        {
            "arg": "--gradient-accumulation",
            "help": "Gradient accumulation steps",
            "required": False,
            "default": 1,
            "type": int,
            "alias": ["--gradient-accumulation"],
        },
        {
            "arg": "--disable-gradient-checkpointing",
            "help": "Disable gradient checkpointing",
            "required": False,
            "action": "store_true",
            "alias": ["--disable-gradient-checkpointing", "--disable-gc"],
        },
        {
            "arg": "--lr",
            "help": "Learning rate",
            "required": False,
            "default": 5e-4,
            "type": float,
        },
        {
            "arg": "--log",
            "help": "Use experiment tracking",
            "required": False,
            "type": str,
            "default": "none",
            "choices": ["none", "wandb", "tensorboard"],
        },
    ]
    return args


def python_type_from_schema_field(field_data: dict) -> Type:
    """Converts JSON schema field types to Python types."""
    type_map = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
    }
    field_type = field_data.get("type")
    if field_type:
        return type_map.get(field_type, str)
    elif "anyOf" in field_data:
        for type_option in field_data["anyOf"]:
            if type_option["type"] != "null":
                return type_map.get(type_option["type"], str)
    return str


def get_default_value(field_data: dict) -> Any:
    return field_data["default"]


def get_field_info(params_class):
    schema = params_class.model_json_schema()
    properties = schema.get("properties", {})
    # field_info = [
    #     {
    #         "arg": f"--{field_name.replace('_', '-')}",
    #         "type": python_type_from_schema_field(field_data),
    #         "help": field_data.get("title", ""),
    #         "default": get_default_value(field_data),
    #     }
    #     for field_name, field_data in properties.items()
    # ]
    field_info = []
    for field_name, field_data in properties.items():
        temp_info = {
            "arg": f"--{field_name.replace('_', '-')}",
            "type": python_type_from_schema_field(field_data),
            "help": field_data.get("title", ""),
            "default": get_default_value(field_data),
        }
        if temp_info["type"] == bool:
            temp_info["action"] = "store_true"

        field_info.append(temp_info)
    return field_info


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
        params.text_column = "autotrain_text"
        params.target_column = "autotrain_label"
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


def dreambooth_munge_data(params, local):
    # check if params.image_path is a directory
    if os.path.isdir(params.image_path):
        training_data = [os.path.join(params.image_path, f) for f in os.listdir(params.image_path)]
        dset = AutoTrainDreamboothDataset(
            concept_images=training_data,
            concept_name=params.prompt,
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            local=local,
        )
        params.image_path = dset.prepare()
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
