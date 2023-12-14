import os

import torch

from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.generic.params import GenericParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams


def launch_command(params):
    num_gpus = torch.cuda.device_count()
    if isinstance(params, LLMTrainingParams):
        if num_gpus == 0:
            raise ValueError("No GPU found. Please use a GPU instance.")
        if num_gpus == 1:
            cmd = [
                "accelerate",
                "launch",
                "--num_machines",
                "1",
                "--num_processes",
                "1",
            ]
        else:
            if params.quantization in ("int8", "int4") and params.peft:
                cmd = [
                    "accelerate",
                    "launch",
                    "--multi_gpu",
                    "--num_machines",
                    "1",
                    "--num_processes",
                    str(num_gpus),
                ]
            else:
                cmd = [
                    "accelerate",
                    "launch",
                    "--use_deepspeed",
                    "--zero_stage",
                    "3",
                    "--offload_optimizer_device",
                    "none",
                    "--offload_param_device",
                    "none",
                    "--zero3_save_16bit_model",
                    "true",
                ]

        cmd.append("--mixed_precision")
        if params.mixed_precision == "fp16":
            cmd.append("fp16")
        elif params.mixed_precision == "bf16":
            cmd.append("bf16")
        else:
            cmd.append("no")

        cmd.extend(
            [
                "-m",
                "autotrain.trainers.clm",
                "--training_config",
                os.path.join(params.project_name, "training_params.json"),
            ]
        )
    elif isinstance(params, DreamBoothTrainingParams):
        cmd = [
            "python",
            "-m",
            "autotrain.trainers.dreambooth",
            "--training_config",
            os.path.join(params.project_name, "training_params.json"),
        ]
    elif isinstance(params, GenericParams):
        cmd = [
            "python",
            "-m",
            "autotrain.trainers.generic",
            "--config",
            os.path.join(params.project_name, "training_params.json"),
        ]
    elif isinstance(params, TabularParams):
        cmd = [
            "python",
            "-m",
            "autotrain.trainers.tabular",
            "--training_config",
            os.path.join(params.project_name, "training_params.json"),
        ]
    elif isinstance(params, TextClassificationParams):
        if num_gpus == 0:
            cmd = [
                "accelerate",
                "launch",
                "--cpu",
            ]
        elif num_gpus == 1:
            cmd = [
                "accelerate",
                "launch",
                "--num_machines",
                "1",
                "--num_processes",
                "1",
            ]
        else:
            cmd = [
                "accelerate",
                "launch",
                "--multi_gpu",
                "--num_machines",
                "1",
                "--num_processes",
                str(num_gpus),
            ]

        cmd.append("--mixed_precision")
        if params.mixed_precision == "fp16":
            cmd.append("fp16")
        elif params.mixed_precision == "bf16":
            cmd.append("bf16")
        else:
            cmd.append("no")

        cmd.extend(
            [
                "-m",
                "autotrain.trainers.text_classification",
                "--training_config",
                os.path.join(params.project_name, "training_params.json"),
            ]
        )
    elif isinstance(params, ImageClassificationParams):
        if num_gpus == 0:
            cmd = [
                "accelerate",
                "launch",
                "--cpu",
            ]
        elif num_gpus == 1:
            cmd = [
                "accelerate",
                "launch",
                "--num_machines",
                "1",
                "--num_processes",
                "1",
            ]
        else:
            cmd = [
                "accelerate",
                "launch",
                "--multi_gpu",
                "--num_machines",
                "1",
                "--num_processes",
                str(num_gpus),
            ]

        cmd.append("--mixed_precision")
        if params.mixed_precision == "fp16":
            cmd.append("fp16")
        elif params.mixed_precision == "bf16":
            cmd.append("bf16")
        else:
            cmd.append("no")

        cmd.extend(
            [
                "-m",
                "autotrain.trainers.image_classification",
                "--training_config",
                os.path.join(params.project_name, "training_params.json"),
            ]
        )
    elif isinstance(params, Seq2SeqParams):
        if num_gpus == 0:
            raise ValueError("No GPU found. Please use a GPU instance.")
        if num_gpus == 1:
            cmd = [
                "accelerate",
                "launch",
                "--num_machines",
                "1",
                "--num_processes",
                "1",
            ]
        else:
            if params.quantization in ("int8", "int4") and params.peft:
                cmd = [
                    "accelerate",
                    "launch",
                    "--multi_gpu",
                    "--num_machines",
                    "1",
                    "--num_processes",
                    str(num_gpus),
                ]
            else:
                cmd = [
                    "accelerate",
                    "launch",
                    "--use_deepspeed",
                    "--zero_stage",
                    "3",
                    "--offload_optimizer_device",
                    "none",
                    "--offload_param_device",
                    "none",
                    "--zero3_save_16bit_model",
                    "true",
                ]
        cmd.append("--mixed_precision")
        if params.mixed_precision == "fp16":
            cmd.append("fp16")
        elif params.mixed_precision == "bf16":
            cmd.append("bf16")
        else:
            cmd.append("no")

        cmd.extend(
            [
                "-m",
                "autotrain.trainers.seq2seq",
                "--training_config",
                os.path.join(params.project_name, "training_params.json"),
            ]
        )

    else:
        raise ValueError("Unsupported params type")

    return cmd
