import io
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Union

import requests
from huggingface_hub import HfApi

from autotrain import logger
from autotrain.app_utils import run_training
from autotrain.dataset import AutoTrainDataset, AutoTrainDreamboothDataset
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.generic.params import GenericParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams


_DOCKERFILE = """
FROM huggingface/autotrain-advanced:latest

CMD autotrain api --port 7860 --host 0.0.0.0
"""

# format _DOCKERFILE
_DOCKERFILE = _DOCKERFILE.replace("\n", " ").replace("  ", "\n").strip()


def _tabular_munge_data(params, username):
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

    train_data_path = f"{params.data_path}/{params.train_split}.csv"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.csv"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            task=task,
            token=params.token,
            project_name=params.project_name,
            username=username,
            column_mapping={"id": params.col_map_id, "label": col_map_label},
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            percent_valid=None,  # TODO: add to UI
        )
        dset.prepare()
        return f"{username}/autotrain-data-{params.project_name}"

    return params.data_path


def _llm_munge_data(params, username):
    train_data_path = f"{params.data_path}/{params.train_split}.csv"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.csv"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        col_map = {"text": params.text_column}
        if params.rejected_text_column is not None:
            col_map["rejected_text"] = params.rejected_text_column
        if params.prompt_column is not None:
            col_map["prompt"] = params.prompt_column
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            task="lm_training",
            token=params.token,
            project_name=params.project_name,
            username=username,
            column_mapping=col_map,
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            percent_valid=None,  # TODO: add to UI
        )
        dset.prepare()
        return f"{username}/autotrain-data-{params.project_name}"

    return params.data_path


def _seq2seq_munge_data(params, username):
    train_data_path = f"{params.data_path}/{params.train_split}.csv"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.csv"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            task="seq2seq",
            token=params.token,
            project_name=params.project_name,
            username=username,
            column_mapping={"text": params.text_column, "label": params.target_column},
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            percent_valid=None,  # TODO: add to UI
        )
        dset.prepare()
        return f"{username}/autotrain-data-{params.project_name}"

    return params.data_path


def _text_clf_munge_data(params, username):
    train_data_path = f"{params.data_path}/{params.train_split}.csv"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.csv"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            task="text_multi_class_classification",
            token=params.token,
            project_name=params.project_name,
            username=username,
            column_mapping={"text": params.text_column, "label": params.target_column},
            percent_valid=None,  # TODO: add to UI
        )
        dset.prepare()
        return f"{username}/autotrain-data-{params.project_name}"

    return params.data_path


def _img_clf_munge_data(params, username):
    train_data_path = f"{params.data_path}/{params.train_split}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}"
    else:
        valid_data_path = None
    if os.path.isdir(train_data_path) or os.path.isdir(valid_data_path):
        raise Exception("Image classification is not yet supported for local datasets.")
    return params.data_path


def _dreambooth_munge_data(params, username):
    # check if params.image_path is a directory
    if os.path.isdir(params.image_path):
        training_data = [os.path.join(params.image_path, f) for f in os.listdir(params.image_path)]
        training_data = [io.BytesIO(open(f, "rb").read()) for f in training_data]
        dset = AutoTrainDreamboothDataset(
            concept_images=training_data,
            concept_name=params.prompt,
            token=params.token,
            project_name=params.project_name,
            username=username,
        )
        dset.prepare()
        return f"{username}/autotrain-data-{params.project_name}"
    return params.image_path


@dataclass
class SpaceRunner:
    params: Union[
        TextClassificationParams,
        ImageClassificationParams,
        LLMTrainingParams,
        GenericParams,
        TabularParams,
        DreamBoothTrainingParams,
        Seq2SeqParams,
    ]
    backend: str

    def __post_init__(self):
        self.spaces_backends = {
            "a10gl": "a10g-large",
            "a10gs": "a10g-small",
            "a100": "a100-large",
            "t4m": "t4-medium",
            "t4s": "t4-small",
            "cpu": "cpu-upgrade",
            "cpuf": "cpu-basic",
            "dgx-a100": "dgx-ngc",
            "ep-aws-useast1-s": "aws_us-east-1_gpu_small_g4dn.xlarge",
            "ep-aws-useast1-m": "aws_us-east-1_gpu_medium_g5.2xlarge",
            "ep-aws-useast1-l": "aws_us-east-1_gpu_large_g4dn.12xlarge",
            "ep-aws-useast1-xl": "aws_us-east-1_gpu_xlarge_p4de",
            "ep-aws-useast1-2xl": "aws_us-east-1_gpu_2xlarge_p4de",
            "ep-aws-useast1-4xl": "aws_us-east-1_gpu_4xlarge_p4de",
            "ep-aws-useast1-8xl": "aws_us-east-1_gpu_8xlarge_p4de",
        }

        if not isinstance(self.params, GenericParams):
            if self.params.repo_id is not None:
                self.username = self.params.repo_id.split("/")[0]
            elif self.params.username is not None:
                self.username = self.params.username
            else:
                raise ValueError("Must provide either repo_id or username")
        else:
            self.username = self.params.username

        self.ep_api_url = f"https://api.endpoints.huggingface.cloud/v2/endpoint/{self.username}"

        if self.params.repo_id is None and self.params.username is not None:
            self.params.repo_id = f"{self.params.username}/{self.params.project_name}"

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
        else:
            raise NotImplementedError

    def prepare(self):
        if isinstance(self.params, LLMTrainingParams):
            self.task_id = 9
            data_path = _llm_munge_data(self.params, self.username)
            self.params.data_path = data_path
            space_id = self._create_space()
            return space_id
        if isinstance(self.params, TextClassificationParams):
            self.task_id = 2
            data_path = _text_clf_munge_data(self.params, self.username)
            self.params.data_path = data_path
            space_id = self._create_space()
            return space_id
        if isinstance(self.params, TabularParams):
            self.task_id = 26
            data_path = _tabular_munge_data(self.params, self.username)
            self.params.data_path = data_path
            space_id = self._create_space()
            return space_id
        if isinstance(self.params, GenericParams):
            self.task_id = 27
            space_id = self._create_space()
            return space_id
        if isinstance(self.params, DreamBoothTrainingParams):
            self.task_id = 25
            data_path = _dreambooth_munge_data(self.params, self.username)
            space_id = self._create_space()
            return space_id
        if isinstance(self.params, Seq2SeqParams):
            self.task_id = 28
            data_path = _seq2seq_munge_data(self.params, self.username)
            space_id = self._create_space()
            return space_id
        if isinstance(self.params, ImageClassificationParams):
            self.task_id = 18
            data_path = _img_clf_munge_data(self.params, self.username)
            space_id = self._create_space()
            return space_id
        raise NotImplementedError

    def _create_readme(self):
        _readme = "---\n"
        _readme += f"title: {self.params.project_name}\n"
        _readme += "emoji: 🚀\n"
        _readme += "colorFrom: green\n"
        _readme += "colorTo: indigo\n"
        _readme += "sdk: docker\n"
        _readme += "pinned: false\n"
        _readme += "duplicated_from: autotrain-projects/autotrain-advanced\n"
        _readme += "---\n"
        _readme = io.BytesIO(_readme.encode())
        return _readme

    def _add_secrets(self, api, repo_id):
        if isinstance(self.params, GenericParams):
            for k, v in self.params.env.items():
                api.add_space_secret(repo_id=repo_id, key=k, value=v)
            self.params.env = {}

        api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value=self.params.token)
        api.add_space_secret(repo_id=repo_id, key="AUTOTRAIN_USERNAME", value=self.username)
        api.add_space_secret(repo_id=repo_id, key="PROJECT_NAME", value=self.params.project_name)
        api.add_space_secret(repo_id=repo_id, key="TASK_ID", value=str(self.task_id))
        api.add_space_secret(repo_id=repo_id, key="PARAMS", value=self.params.model_dump_json())

        if isinstance(self.params, DreamBoothTrainingParams):
            api.add_space_secret(repo_id=repo_id, key="DATA_PATH", value=self.params.image_path)
        else:
            api.add_space_secret(repo_id=repo_id, key="DATA_PATH", value=self.params.data_path)

        if not isinstance(self.params, GenericParams):
            api.add_space_secret(repo_id=repo_id, key="MODEL", value=self.params.model)
            api.add_space_secret(repo_id=repo_id, key="OUTPUT_MODEL_REPO", value=self.params.repo_id)

    def _create_endpoint(self):
        hardware = self.spaces_backends[self.backend]
        accelerator = hardware.split("_")[2]
        instance_size = hardware.split("_")[3]
        region = hardware.split("_")[1]
        vendor = hardware.split("_")[0]
        instance_type = hardware.split("_")[4]
        payload = {
            "accountId": self.username,
            "compute": {
                "accelerator": accelerator,
                "instanceSize": instance_size,
                "instanceType": instance_type,
                "scaling": {"maxReplica": 1, "minReplica": 1},
            },
            "model": {
                "framework": "custom",
                "image": {
                    "custom": {
                        "env": {
                            "HF_TOKEN": self.params.token,
                            "AUTOTRAIN_USERNAME": self.username,
                            "PROJECT_NAME": self.params.project_name,
                            "PARAMS": self.params.model_dump_json(),
                            "DATA_PATH": self.params.data_path,
                            "TASK_ID": str(self.task_id),
                            "MODEL": self.params.model,
                            "OUTPUT_MODEL_REPO": self.params.repo_id,
                            "ENDPOINT_ID": f"{self.username}/{self.params.project_name}",
                        },
                        "health_route": "/",
                        "port": 7860,
                        "url": "public.ecr.aws/z4c3o6n6/autotrain-api:latest",
                    }
                },
                "repository": "autotrain-projects/autotrain-advanced",
                "revision": "main",
                "task": "custom",
            },
            "name": self.params.project_name,
            "provider": {"region": region, "vendor": vendor},
            "type": "protected",
        }
        headers = {"Authorization": f"Bearer {self.params.token}"}
        r = requests.post(self.ep_api_url, json=payload, headers=headers, timeout=120)
        return r.json()["name"]

    def _create_space(self):
        if self.backend.startswith("dgx-") or self.backend == "local":
            env_vars = {
                "HF_TOKEN": self.params.token,
                "AUTOTRAIN_USERNAME": self.username,
                "PROJECT_NAME": self.params.project_name,
                "TASK_ID": str(self.task_id),
                "PARAMS": json.dumps(self.params.model_dump_json()),
            }
            if isinstance(self.params, DreamBoothTrainingParams):
                env_vars["DATA_PATH"] = self.params.image_path
            else:
                env_vars["DATA_PATH"] = self.params.data_path

            if not isinstance(self.params, GenericParams):
                env_vars["MODEL"] = self.params.model
                env_vars["OUTPUT_MODEL_REPO"] = self.params.repo_id

            if self.backend.startswith("dgx-"):
                ngc_runner = NGCRunner(
                    job_name=self.params.repo_id.replace("/", "-"),
                    env_vars=env_vars,
                    backend=self.backend,
                )
                ngc_runner.create()
                return
            else:
                local_runner = LocalRunner(env_vars=env_vars)
                pid = local_runner.create()
                return pid

        if self.backend.startswith("ep-"):
            endpoint_id = self._create_endpoint()
            return endpoint_id

        api = HfApi(token=self.params.token)
        repo_id = f"{self.username}/autotrain-{self.params.project_name}"
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            space_hardware=self.spaces_backends[self.backend.split("-")[1].lower()],
            private=True,
        )
        self._add_secrets(api, repo_id)
        readme = self._create_readme()
        api.upload_file(
            path_or_fileobj=readme,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="space",
        )

        _dockerfile = io.BytesIO(_DOCKERFILE.encode())
        api.upload_file(
            path_or_fileobj=_dockerfile,
            path_in_repo="Dockerfile",
            repo_id=repo_id,
            repo_type="space",
        )
        return repo_id


@dataclass
class LocalRunner:
    env_vars: dict

    def create(self):
        logger.info("Starting server")
        params = self.env_vars["PARAMS"]
        task_id = int(self.env_vars["TASK_ID"])
        training_pid = run_training(params, task_id, local=True)
        return training_pid


@dataclass
class NGCRunner:
    job_name: str
    env_vars: dict
    backend: str
    enable_diag: bool = False

    def __post_init__(self):
        self.ngc_ace = os.environ.get("NGC_ACE")
        self.ngc_org = os.environ.get("NGC_ORG")
        self.ngc_api_key = os.environ.get("NGC_CLI_API_KEY")
        self.ngc_team = os.environ.get("NGC_TEAM")
        self.instance_map = {
            "dgx-a100": "dgxa100.80g.1.norm",
            "dgx-2a100": "dgxa100.80g.2.norm",
            "dgx-4a100": "dgxa100.80g.4.norm",
            "dgx-8a100": "dgxa100.80g.8.norm",
        }
        logger.info("Creating NGC Job")
        logger.info(f"NGC_ACE: {self.ngc_ace}")
        logger.info(f"NGC_ORG: {self.ngc_org}")
        logger.info(f"job_name: {self.job_name}")
        logger.info(f"backend: {self.backend}")

    def create(self):
        cmd = "ngc base-command job run --name {job_name}"
        cmd += " --priority NORMAL --order 50 --preempt RUNONCE --min-timeslice 0s"
        cmd += " --total-runtime 259200s --ace {ngc_ace} --org {ngc_org} --instance {instance}"
        cmd += " --commandline 'set -x; conda run --no-capture-output -p /app/env autotrain api --port 7860 --host 0.0.0.0' -p 7860 --result /results"
        cmd += " --image '{ngc_org}/autotrain-advanced:latest'"

        cmd = cmd.format(
            job_name=self.job_name,
            ngc_ace=self.ngc_ace,
            ngc_org=self.ngc_org,
            instance=self.instance_map[self.backend],
        )

        for k, v in self.env_vars.items():
            cmd += f" --env-var {k}:{v}"

        ngc_config_cmd = "ngc config set"
        ngc_config_cmd += " --team {ngc_team} --org {ngc_org} --ace {ngc_ace}"
        ngc_config_cmd = ngc_config_cmd.format(
            # ngc_api_key=self.ngc_api_key,
            ngc_team=self.ngc_team,
            ngc_org=self.ngc_org,
            ngc_ace=self.ngc_ace,
        )
        logger.info("Setting NGC API key")
        ngc_config_process = subprocess.Popen(ngc_config_cmd, shell=True)
        ngc_config_process.wait()

        if ngc_config_process.returncode == 0:
            logger.info("NGC API key set successfully")
        else:
            logger.error("Failed to set NGC API key")
            # print full output
            logger.error(ngc_config_process.stdout.read())
            logger.error(ngc_config_process.stderr.read())
            raise Exception("Failed to set NGC API key")

        if self.enable_diag:
            ngc_diag_cmd = ["ngc", "diag", "all"]
            process = subprocess.run(ngc_diag_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = process.stdout
            error = process.stderr
            if process.returncode != 0:
                logger.info("NGC DIAG ALL Error occurred:")
                logger.info(error)
            else:
                logger.info("NGC DIAG ALL output:")
                logger.info(output)

        logger.info("Creating NGC Job")
        subprocess.run(
            cmd,
            shell=True,
            check=True,
        )
