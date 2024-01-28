import base64
import io
import json
import os
import subprocess
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Union

import requests
from huggingface_hub import HfApi
from requests.exceptions import HTTPError

from autotrain import logger
from autotrain.app_utils import run_training
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
            "nvcf-a100": "nvcf-ngc",
            "local": "local",
            "local-cli": "local-cli",
            "ep-aws-useast1-s": "aws_us-east-1_gpu_small_g4dn.xlarge",
            "ep-aws-useast1-m": "aws_us-east-1_gpu_medium_g5.2xlarge",
            "ep-aws-useast1-l": "aws_us-east-1_gpu_large_g4dn.12xlarge",
            "ep-aws-useast1-xl": "aws_us-east-1_gpu_xlarge_p4de",
            "ep-aws-useast1-2xl": "aws_us-east-1_gpu_2xlarge_p4de",
            "ep-aws-useast1-4xl": "aws_us-east-1_gpu_4xlarge_p4de",
            "ep-aws-useast1-8xl": "aws_us-east-1_gpu_8xlarge_p4de",
        }

        if not isinstance(self.params, GenericParams) and self.backend != "local-cli":
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
            space_id = self._create_space()
            return space_id
        if isinstance(self.params, TextClassificationParams):
            self.task_id = 2
            space_id = self._create_space()
            return space_id
        if isinstance(self.params, TabularParams):
            self.task_id = 26
            space_id = self._create_space()
            return space_id
        if isinstance(self.params, GenericParams):
            self.task_id = 27
            space_id = self._create_space()
            return space_id
        if isinstance(self.params, DreamBoothTrainingParams):
            self.task_id = 25
            space_id = self._create_space()
            return space_id
        if isinstance(self.params, Seq2SeqParams):
            self.task_id = 28
            space_id = self._create_space()
            return space_id
        if isinstance(self.params, ImageClassificationParams):
            self.task_id = 18
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
        if self.backend.startswith("dgx-") or self.backend.startswith("nvcf-") or self.backend.startswith("local"):
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
                if int(os.environ.get("NGC_USE_CLI", 0)) == 1:
                    ngc_runner.create_cli()
                else:
                    ngc_runner.create()
                return
            elif self.backend.startswith("nvcf-"):
                nvcf_runner = NVCFRunner(
                    job_name=self.params.repo_id.replace("/", "-"),
                    env_vars=env_vars,
                    backend=self.backend,
                )
                nvcf_runner.create()
                return
            else:
                local_runner = LocalRunner(env_vars=env_vars, wait=self.backend == "local-cli")
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
    wait: bool = False

    def create(self):
        logger.info("Starting local training...")
        params = self.env_vars["PARAMS"]
        task_id = int(self.env_vars["TASK_ID"])
        training_pid = run_training(params, task_id, local=True, wait=self.wait)
        if not self.wait:
            logger.info(f"Training PID: {training_pid}")
        return training_pid


@dataclass
class NGCRunner:
    job_name: str
    env_vars: dict
    backend: str
    enable_diag: bool = False

    def __post_init__(self):
        self.ngc_api = os.environ.get("NGC_API", "https://api.ngc.nvidia.com/v2/org")
        self.ngc_auth = os.environ.get("NGC_AUTH", "https://authn.nvidia.com")

        self.env_vars["BACKEND"] = self.backend

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

    def _user_authentication_ngc(self):
        logger.info("Authenticating NGC user...")
        scope = "group/ngc"

        querystring = {"service": "ngc", "scope": scope}
        auth = f"$oauthtoken:{self.ngc_api_key}"
        headers = {
            "Authorization": f"Basic {base64.b64encode(auth.encode('utf-8')).decode('utf-8')}",
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
        }
        try:
            response = requests.get(self.ngc_auth + "/token", headers=headers, params=querystring, timeout=30)
        except HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            raise Exception("HTTP Error %d: from '%s'" % (response.status_code, self.ngc_auth))
        except (requests.Timeout, ConnectionError) as err:
            logger.error(f"Failed to request NGC token - {repr(err)}")
            raise Exception("%s is unreachable, please try again later." % self.ngc_auth)
        return json.loads(response.text.encode("utf8"))["token"]

    def _create_ngc_job(self, token, url, payload):
        logger.info("Creating NGC Job")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        try:
            response = requests.post(self.ngc_api + url + "/jobs", headers=headers, json=payload, timeout=30)
            result = response.json()
            logger.info(
                f"NGC Job ID: {result.get('job', {}).get('id')}, Job Status History: {result.get('jobStatusHistory')}"
            )

        except HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            raise Exception(f"HTTP Error {response.status_code}: {http_err}")
        except (requests.Timeout, ConnectionError) as err:
            logger.error(f"Failed to create NGC job - {repr(err)}")
            raise Exception(f"Unreachable, please try again later: {err}")
        return json.loads(response.text.encode("utf8"))

    def create(self):
        ngc_url = f"/{self.ngc_org}/team/{self.ngc_team}"
        ngc_cmd = "set -x; conda run --no-capture-output -p /app/env autotrain api --port 7860 --host 0.0.0.0"
        ngc_payload = {
            "name": self.job_name,
            "aceName": self.ngc_ace,
            "aceInstance": self.instance_map[self.backend],
            "dockerImageName": f"{self.ngc_org}/autotrain-advanced:latest",
            "command": ngc_cmd,
            "envs": [{"name": key, "value": value} for key, value in self.env_vars.items()],
            "jobOrder": 50,
            "jobPriority": "NORMAL",
            "portMappings": [{"containerPort": 7860, "protocol": "HTTPS"}],
            "resultContainerMountPoint": "/results",
            "runPolicy": {"preemptClass": "RUNONCE", "totalRuntimeSeconds": 259200},
        }

        ngc_token = self._user_authentication_ngc()
        self._create_ngc_job(ngc_token, ngc_url, ngc_payload)

    def create_cli(self):
        cmd = "ngc base-command job run --name {job_name}"
        cmd += " --priority NORMAL --order 50 --preempt RUNONCE --min-timeslice 0s"
        cmd += " --total-runtime 259200s --instance {instance}"
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


@dataclass
class NVCFRunner:
    job_name: str
    env_vars: dict
    backend: str

    def __post_init__(self):
        self.nvcf_api = os.environ.get("NVCF_API")
        self.nvcf_backend = os.environ.get("NVCF_BACKEND")
        self.nvcf_image = os.environ.get("NVCF_IMAGE")
        self.nvcf_nca_id = os.environ.get("NVCF_NCA_ID")
        self.nvcf_jwt_provider = os.environ.get("NVCF_JWT_PROVIDER")
        self.nvcf_ssa_client_id = os.environ.get("NVCF_SSA_CLIENT_ID")
        self.nvcf_ssa_client_secret = os.environ.get("NVCF_SSA_CLIENT_SECRET")

        self.instance_map = {
            "nvcf-a100": {"gpu": "A100_80GB", "instanceType": "BM.GPU.A100-v2.8"},
            "nvcf-8a100": {"gpu": "A100_80GB_8GPU", "instanceType": "BM.GPU.A100-v2.8_8x"},
        }

        logger.info("Starting NVCF training")
        logger.info(f"job_name: {self.job_name}")
        logger.info(f"backend: {self.backend}")

    def _convert_dict_to_object(self, dictionary):
        if isinstance(dictionary, dict):
            for key, value in dictionary.items():
                dictionary[key] = self._convert_dict_to_object(value)
            return SimpleNamespace(**dictionary)
        elif isinstance(dictionary, list):
            return [self._convert_dict_to_object(item) for item in dictionary]
        else:
            return dictionary

    def _user_authentication_nvcf(self):
        logger.info("Authenticating NVCF client...")
        auth = base64.b64encode(f"{self.nvcf_ssa_client_id}:{self.nvcf_ssa_client_secret}".encode()).decode()
        logger.info(f"{auth}")

        headers = {"Authorization": f"Basic {auth}", "Content-Type": "application/x-www-form-urlencoded"}
        body = {
            "grant_type": "client_credentials",
            "scope": "register_function update_function delete_function list_functions deploy_function invoke_function queue_details authorize_clients",
        }
        try:
            logger.info(f"{body} -- {headers}")
            response = requests.post(self.nvcf_jwt_provider + "/token", headers=headers, data=body, timeout=30)
            logger.info(response.__dict__)

            if response.status_code == 200:
                resp_data = response.json()
                bearer_token = resp_data.get("access_token")
                return bearer_token
            else:
                raise Exception(
                    f"Failed to get JWT token: {response.status_code} - {response.headers} - {response.text}"
                )
        except HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            raise Exception("HTTP Error %d: from '%s'" % (response.status_code, self.ngc_auth))
        except (requests.Timeout, ConnectionError) as err:
            logger.error(f"Failed to request NGC token - {repr(err)}")
            raise Exception("%s is unreachable, please try again later." % self.ngc_auth)

    def _create_nvcf(self, token, nvcf_type, url, payload):
        logger.info(f"Configuring NVCF {nvcf_type}")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            logger.info(f"NVCF response: {response.__dict__}")
            result = response.json()
            result_obj = self._convert_dict_to_object(result)
            logger.info(f"NVCF {nvcf_type}: {result_obj}")
            return result_obj
        except HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            raise Exception(f"HTTP Error {response.status_code}: {http_err}")
        except (requests.Timeout, ConnectionError) as err:
            logger.error(f"Failed to create NGC job - {repr(err)}")
            raise Exception(f"Unreachable, please try again later: {err}")

    def _poll_nvcf_deploy(self, url, token, timeout=600, interval=10):
        timeout = float(timeout)
        interval = float(interval)
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                function_status = data.get("deployment", {}).get("functionStatus", "")
                if function_status == "ACTIVE":
                    elapsed_time = time.time() - start_time
                    logger.info(f"Function activated in {elapsed_time:.2f} seconds.")
                    return function_status
                else:
                    logger.info("Waiting for active function...")
                time.sleep(interval)  # Wait for a specified interval before the next poll
            except requests.HTTPError as http_err:
                raise Exception(f"HTTP error occurred: {http_err}")
            except requests.RequestException as err:
                raise Exception(f"Error while requesting URL: {err}")
            except ValueError:
                raise Exception("Error parsing JSON response")

        raise TimeoutError(f"Timeout reached: Function did not become ACTIVE within {timeout} seconds")

    def _poll_nvcf_train(self, url, token, timeout=86400, interval=30):
        timeout = float(timeout)
        interval = float(interval)
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                function_status = data.get("deployment", {}).get("functionStatus", "")
                if function_status == "ACTIVE":
                    elapsed_time = time.time() - start_time
                    logger.info(f"Function activated in {elapsed_time:.2f} seconds.")
                    return function_status
                else:
                    logger.info("Waiting for active function...")
                time.sleep(interval)  # Wait for a specified interval before the next poll
            except requests.HTTPError as http_err:
                raise Exception(f"HTTP error occurred: {http_err}")
            except requests.RequestException as err:
                raise Exception(f"Error while requesting URL: {err}")
            except ValueError:
                raise Exception("Error parsing JSON response")

        raise TimeoutError(f"Timeout reached: Function did not become ACTIVE within {timeout} seconds")

    def create(self):
        nvcf_url = f"{self.nvcf_api}/v2/nvcf"
        nvcf_fr_payload = {
            "name": self.job_name,
            "inferenceUrl": "job",
            "inferencePort": 7860,
            "healthUri": "health",
            "containerImage": self.nvcf_image,
            "containerEnvironment": [{"key": key, "value": value} for key, value in self.env_vars.items()],
            "apiBodyFormat": "CUSTOM",
        }
        nvcf_fd_payload = {
            "deploymentSpecifications": [
                {
                    "gpu": self.instance_map[self.backend]["gpu"],
                    "instanceType": self.instance_map[self.backend]["instanceType"],
                    "backend": self.nvcf_backend,
                    "maxInstances": 1,
                    "minInstances": 1,
                }
            ]
        }

        nvcf_token = self._user_authentication_nvcf()

        logger.info(f"{nvcf_token} -- {nvcf_url}/functions")
        logger.info(nvcf_fr_payload)

        nvcf_fn = self._create_nvcf(nvcf_token, "function", f"{nvcf_url}/functions", nvcf_fr_payload)
        nvcf_fn_url = f"{nvcf_url}/deployments/functions/{nvcf_fn.function.id}/versions/{nvcf_fn.function.versionId}"
        logger.info(f"Initializing deployment for: {nvcf_fn_url}")
        # logger.info(f"{nvcf_fd_payload}")
        time.sleep(2)
        self._create_nvcf(nvcf_token, "deployment", nvcf_fn_url, nvcf_fd_payload)

        time.sleep(2)
        self._poll_nvcf_deploy(url=nvcf_fn_url, token=nvcf_token)
