import json
import os
import subprocess

from autotrain.commands import launch_command
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.generic.params import GenericParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.image_regression.params import ImageRegressionParams
from autotrain.trainers.object_detection.params import ObjectDetectionParams
from autotrain.trainers.sent_transformers.params import SentenceTransformersParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams, TextClassificationGaudiParams
from autotrain.trainers.text_regression.params import TextRegressionParams
from autotrain.trainers.token_classification.params import TokenClassificationParams


ALLOW_REMOTE_CODE = os.environ.get("ALLOW_REMOTE_CODE", "true").lower() == "true"

# centralize guadi params for other tasks
WHITE_LIST_GAUDI_PARAMS = [
    "model_name_or_path",
    "backend",
    "dataset_name",
    "train_split",
    "valid_split",
    "column_mapping_text_column",
    "column_mapping_target_column",
    "max_seq_length",
    "num_train_epochs",
    "per_device_train_batch_size",
    "learning_rate",
    "optim",
    "lr_scheduler_type",
    "gradient_accumulation_steps",
    "mixed_precision",
    "use_habana",
    "use_hpu_graphs",
    "use_hpu_graphs_for_training",
    "use_hpu_graphs_for_inference",
    "non_blocking_data_copy",
    "evaluation_strategy",
    "username",
    "token",
    "output_dir",
    "push_to_hub",
    "warmup_ratio",
    "weight_decay",
    "max_grad_norm",
    "seed",
    "logging_steps",
    "auto_find_batch_size",
    "save_total_limit",
]


def run_training(params, task_id, local=False, wait=False):
    print("params: ", params)
    params = json.loads(params)
    print("params: ", params)
    project_name = 'test-project'
    print(f"Running training for {project_name}")
    if isinstance(params, str):
        params = json.loads(params)
    if task_id == 9:
        params = LLMTrainingParams(**params)
    elif task_id == 28:
        params = Seq2SeqParams(**params)
    elif task_id in (1, 2):
        # params = {k: v for k, v in params.items() if k not in WHITE_LIST_GAUDI_PARAMS}
        # params = TextClassificationParams(**params)
        print("TextClassificationParams---------")
        print(params)
        # params = {k: v for k, v in params.items() if k not in WHITE_LIST_GAUDI_PARAMS}
        params = TextClassificationGaudiParams(**params)
        print("TextClassificationParams---------")
        print(params)
    elif task_id in (13, 14, 15, 16, 26):
        params = TabularParams(**params)
    elif task_id == 27:
        params = GenericParams(**params)
    elif task_id == 25:
        params = DreamBoothTrainingParams(**params)
    elif task_id == 18:
        params = ImageClassificationParams(**params)
    elif task_id == 4:
        params = TokenClassificationParams(**params)
    elif task_id == 10:
        params = TextRegressionParams(**params)
    elif task_id == 29:
        params = ObjectDetectionParams(**params)
    elif task_id == 30:
        params = SentenceTransformersParams(**params)
    elif task_id == 24:
        params = ImageRegressionParams(**params)
    else:
        raise NotImplementedError

    params.save(output_dir=project_name)
    cmd = launch_command(params=params, project_name=project_name)
    cmd = [str(c) for c in cmd]
    env = os.environ.copy()
    process = subprocess.Popen(cmd, env=env)
    if wait:
        process.wait()
    return process.pid
