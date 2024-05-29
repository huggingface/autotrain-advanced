import json
import os
import subprocess

from autotrain.commands import launch_command
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


ALLOW_REMOTE_CODE = os.environ.get("ALLOW_REMOTE_CODE", "true").lower() == "true"


def run_training(params, task_id, local=False, wait=False):
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
