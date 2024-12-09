import os
import shlex

import torch

from autotrain import logger
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


CPU_COMMAND = [
    "accelerate",
    "launch",
    "--cpu",
]

SINGLE_GPU_COMMAND = [
    "accelerate",
    "launch",
    "--num_machines",
    "1",
    "--num_processes",
    "1",
]


def get_accelerate_command(num_gpus, gradient_accumulation_steps=1, distributed_backend=None):
    """
    Generates the appropriate command to launch a training job using the `accelerate` library based on the number of GPUs
    and the specified distributed backend.

    Args:
        num_gpus (int): The number of GPUs available for training. If 0, training will be forced on CPU.
        gradient_accumulation_steps (int, optional): The number of gradient accumulation steps. Defaults to 1.
        distributed_backend (str, optional): The distributed backend to use. Can be "ddp" (Distributed Data Parallel),
                                             "deepspeed", or None. Defaults to None.

    Returns:
        list or str: The command to be executed as a list of strings. If no GPU is found, returns a CPU command string.
                     If a single GPU is found, returns a single GPU command string. Otherwise, returns a list of
                     command arguments for multi-GPU or DeepSpeed training.

    Raises:
        ValueError: If an unsupported distributed backend is specified.
    """
    if num_gpus == 0:
        logger.warning("No GPU found. Forcing training on CPU. This will be super slow!")
        return CPU_COMMAND

    if num_gpus == 1:
        return SINGLE_GPU_COMMAND

    if distributed_backend in ("ddp", None):
        return [
            "accelerate",
            "launch",
            "--multi_gpu",
            "--num_machines",
            "1",
            "--num_processes",
            str(num_gpus),
        ]
    elif distributed_backend == "deepspeed":
        return [
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
            "--zero3_init_flag",
            "true",
            "--deepspeed_multinode_launcher",
            "standard",
            "--gradient_accumulation_steps",
            str(gradient_accumulation_steps),
        ]
    else:
        raise ValueError("Unsupported distributed backend")


def launch_command(params):
    """
    Launches the appropriate training command based on the type of training parameters provided.

    Args:
        params (object): An instance of one of the training parameter classes. This can be one of the following:
            - LLMTrainingParams
            - GenericParams
            - TabularParams
            - TextClassificationParams
            - TextRegressionParams
            - SentenceTransformersParams
            - ExtractiveQuestionAnsweringParams
            - TokenClassificationParams
            - ImageClassificationParams
            - ObjectDetectionParams
            - ImageRegressionParams
            - Seq2SeqParams
            - VLMTrainingParams

    Returns:
        list: A list of command line arguments to be executed for training.

    Raises:
        ValueError: If the provided params type is unsupported.
    """

    params.project_name = shlex.split(params.project_name)[0]
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    if cuda_available:
        num_gpus = torch.cuda.device_count()
    elif mps_available:
        num_gpus = 1
    else:
        num_gpus = 0
    if isinstance(params, LLMTrainingParams):
        cmd = get_accelerate_command(num_gpus, params.gradient_accumulation, params.distributed_backend)
        if num_gpus > 0:
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
    elif (
        isinstance(params, TextClassificationParams)
        or isinstance(params, TextRegressionParams)
        or isinstance(params, SentenceTransformersParams)
        or isinstance(params, ExtractiveQuestionAnsweringParams)
    ):
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

        if num_gpus > 0:
            cmd.append("--mixed_precision")
            if params.mixed_precision == "fp16":
                cmd.append("fp16")
            elif params.mixed_precision == "bf16":
                cmd.append("bf16")
            else:
                cmd.append("no")

        if isinstance(params, TextRegressionParams):
            cmd.extend(
                [
                    "-m",
                    "autotrain.trainers.text_regression",
                    "--training_config",
                    os.path.join(params.project_name, "training_params.json"),
                ]
            )
        elif isinstance(params, SentenceTransformersParams):
            cmd.extend(
                [
                    "-m",
                    "autotrain.trainers.sent_transformers",
                    "--training_config",
                    os.path.join(params.project_name, "training_params.json"),
                ]
            )
        elif isinstance(params, ExtractiveQuestionAnsweringParams):
            cmd.extend(
                [
                    "-m",
                    "autotrain.trainers.extractive_question_answering",
                    "--training_config",
                    os.path.join(params.project_name, "training_params.json"),
                ]
            )
        else:
            cmd.extend(
                [
                    "-m",
                    "autotrain.trainers.text_classification",
                    "--training_config",
                    os.path.join(params.project_name, "training_params.json"),
                ]
            )
    elif isinstance(params, TokenClassificationParams):
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

        if num_gpus > 0:
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
                "autotrain.trainers.token_classification",
                "--training_config",
                os.path.join(params.project_name, "training_params.json"),
            ]
        )
    elif (
        isinstance(params, ImageClassificationParams)
        or isinstance(params, ObjectDetectionParams)
        or isinstance(params, ImageRegressionParams)
    ):
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

        if num_gpus > 0:
            cmd.append("--mixed_precision")
            if params.mixed_precision == "fp16":
                cmd.append("fp16")
            elif params.mixed_precision == "bf16":
                cmd.append("bf16")
            else:
                cmd.append("no")

        if isinstance(params, ObjectDetectionParams):
            cmd.extend(
                [
                    "-m",
                    "autotrain.trainers.object_detection",
                    "--training_config",
                    os.path.join(params.project_name, "training_params.json"),
                ]
            )
        elif isinstance(params, ImageRegressionParams):
            cmd.extend(
                [
                    "-m",
                    "autotrain.trainers.image_regression",
                    "--training_config",
                    os.path.join(params.project_name, "training_params.json"),
                ]
            )
        else:
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
            logger.warning("No GPU found. Forcing training on CPU. This will be super slow!")
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
        elif num_gpus == 2:
            cmd = [
                "accelerate",
                "launch",
                "--multi_gpu",
                "--num_machines",
                "1",
                "--num_processes",
                "2",
            ]
        else:
            if params.quantization in ("int8", "int4") and params.peft and params.mixed_precision == "bf16":
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
                    "--zero3_init_flag",
                    "true",
                    "--deepspeed_multinode_launcher",
                    "standard",
                    "--gradient_accumulation_steps",
                    str(params.gradient_accumulation),
                ]
        if num_gpus > 0:
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

    elif isinstance(params, VLMTrainingParams):
        if num_gpus == 0:
            logger.warning("No GPU found. Forcing training on CPU. This will be super slow!")
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
        elif num_gpus == 2:
            cmd = [
                "accelerate",
                "launch",
                "--multi_gpu",
                "--num_machines",
                "1",
                "--num_processes",
                "2",
            ]
        else:
            if params.quantization in ("int8", "int4") and params.peft and params.mixed_precision == "bf16":
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
                    "--zero3_init_flag",
                    "true",
                    "--deepspeed_multinode_launcher",
                    "standard",
                    "--gradient_accumulation_steps",
                    str(params.gradient_accumulation),
                ]

        if num_gpus > 0:
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
                "autotrain.trainers.vlm",
                "--training_config",
                os.path.join(params.project_name, "training_params.json"),
            ]
        )

    else:
        raise ValueError("Unsupported params type")

    logger.info(cmd)
    logger.info(params)
    return cmd
