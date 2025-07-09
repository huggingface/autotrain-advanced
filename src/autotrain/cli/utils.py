from typing import Any, Type

from autotrain.backends.base import AVAILABLE_HARDWARE


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
    field_info = []
    for field_name, field_data in properties.items():
        temp_info = {
            "arg": f"--{field_name.replace('_', '-')}",
            "alias": [f"--{field_name}", f"--{field_name.replace('_', '-')}"],
            "type": python_type_from_schema_field(field_data),
            "help": field_data.get("title", ""),
            "default": get_default_value(field_data),
        }
        if temp_info["type"] == bool:
            temp_info["action"] = "store_true"

        field_info.append(temp_info)
    return field_info
