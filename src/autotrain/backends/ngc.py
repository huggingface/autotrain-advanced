import base64
import json
import os

import requests
from requests.exceptions import HTTPError

from autotrain import logger
from autotrain.backends.base import BaseBackend


NGC_API = os.environ.get("NGC_API", "https://api.ngc.nvidia.com/v2/org")
NGC_AUTH = os.environ.get("NGC_AUTH", "https://authn.nvidia.com")
NGC_ACE = os.environ.get("NGC_ACE")
NGC_ORG = os.environ.get("NGC_ORG")
NGC_API_KEY = os.environ.get("NGC_CLI_API_KEY")
NGC_TEAM = os.environ.get("NGC_TEAM")


class NGCRunner(BaseBackend):
    def _user_authentication_ngc(self):
        logger.info("Authenticating NGC user...")
        scope = "group/ngc"

        querystring = {"service": "ngc", "scope": scope}
        auth = f"$oauthtoken:{NGC_API_KEY}"
        headers = {
            "Authorization": f"Basic {base64.b64encode(auth.encode('utf-8')).decode('utf-8')}",
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
        }
        try:
            response = requests.get(NGC_AUTH + "/token", headers=headers, params=querystring, timeout=30)
        except HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            raise Exception("HTTP Error %d: from '%s'" % (response.status_code, NGC_AUTH))
        except (requests.Timeout, ConnectionError) as err:
            logger.error(f"Failed to request NGC token - {repr(err)}")
            raise Exception("%s is unreachable, please try again later." % NGC_AUTH)
        return json.loads(response.text.encode("utf8"))["token"]

    def _create_ngc_job(self, token, url, payload):
        logger.info("Creating NGC Job")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        try:
            response = requests.post(NGC_API + url + "/jobs", headers=headers, json=payload, timeout=30)
            result = response.json()
            logger.info(
                f"NGC Job ID: {result.get('job', {}).get('id')}, Job Status History: {result.get('jobStatusHistory')}"
            )
            return result.get("job", {}).get("id")
        except HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            raise Exception(f"HTTP Error {response.status_code}: {http_err}")
        except (requests.Timeout, ConnectionError) as err:
            logger.error(f"Failed to create NGC job - {repr(err)}")
            raise Exception(f"Unreachable, please try again later: {err}")

    def create(self):
        job_name = f"{self.username}-{self.params.project_name}"
        ngc_url = f"/{NGC_ORG}/team/{NGC_TEAM}"
        ngc_cmd = "set -x; conda run --no-capture-output -p /app/env autotrain api --port 7860 --host 0.0.0.0"
        ngc_payload = {
            "name": job_name,
            "aceName": NGC_ACE,
            "aceInstance": self.available_hardware[self.backend],
            "dockerImageName": f"{NGC_ORG}/autotrain-advanced:latest",
            "command": ngc_cmd,
            "envs": [{"name": key, "value": value} for key, value in self.env_vars.items()],
            "jobOrder": 50,
            "jobPriority": "NORMAL",
            "portMappings": [{"containerPort": 7860, "protocol": "HTTPS"}],
            "resultContainerMountPoint": "/results",
            "runPolicy": {"preemptClass": "RUNONCE", "totalRuntimeSeconds": 259200},
        }

        ngc_token = self._user_authentication_ngc()
        job_id = self._create_ngc_job(ngc_token, ngc_url, ngc_payload)
        return job_id
