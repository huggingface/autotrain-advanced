import json
import os
import subprocess

from autotrain.commands import launch_command
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


ALLOW_REMOTE_CODE = os.environ.get("ALLOW_REMOTE_CODE", "true").lower() == "true"


def run_training(params, task_id, local=False, wait=False):
    """
    Run the training process based on the provided parameters and task ID.

    Args:
        params (str): JSON string of the parameters required for training.
        task_id (int): Identifier for the type of task to be performed.
        local (bool, optional): Flag to indicate if the training should be run locally. Defaults to False.
        wait (bool, optional): Flag to indicate if the function should wait for the process to complete. Defaults to False.

    Returns:
        int: Process ID of the launched training process.

    Raises:
        NotImplementedError: If the task_id does not match any of the predefined tasks.
    """
    params = json.loads(params)
    if isinstance(params, str):
        params = json.loads(params)
    if task_id == 9:
        params = LLMTrainingParams(**params)
    elif task_id == 28:
        params = Seq2SeqParams(**params)
    elif task_id in (1, 2):
        params = TextClassificationParams(**params)
    elif task_id in (13, 14, 15, 16, 26):
        params = TabularParams(**params)
    elif task_id == 27:
        params = GenericParams(**params)
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
    elif task_id == 31:
        params = VLMTrainingParams(**params)
    elif task_id == 5:
        params = ExtractiveQuestionAnsweringParams(**params)
    else:
        raise NotImplementedError

    params.save(output_dir=params.project_name)
    cmd = launch_command(params=params)
    cmd = [str(c) for c in cmd]
    env = os.environ.copy()
    process = subprocess.Popen(cmd, env=env)
    if wait:
        process.wait()
    return process.pid
