import requests

from autotrain.backends.base import BaseBackend


ENDPOINTS_URL = "https://api.endpoints.huggingface.cloud/v2/endpoint/"

AVAILABLE_HARDWARE = {
    "ep-aws-useast1-s": "aws_us-east-1_gpu_small_g4dn.xlarge",
    "ep-aws-useast1-m": "aws_us-east-1_gpu_medium_g5.2xlarge",
    "ep-aws-useast1-l": "aws_us-east-1_gpu_large_g4dn.12xlarge",
    "ep-aws-useast1-xl": "aws_us-east-1_gpu_xlarge_p4de",
    "ep-aws-useast1-2xl": "aws_us-east-1_gpu_2xlarge_p4de",
    "ep-aws-useast1-4xl": "aws_us-east-1_gpu_4xlarge_p4de",
    "ep-aws-useast1-8xl": "aws_us-east-1_gpu_8xlarge_p4de",
}


class EndpointsRunner(BaseBackend):
    def _create(self):
        hardware = AVAILABLE_HARDWARE[self.backend]
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
        r = requests.post(
            ENDPOINTS_URL + self.username,
            json=payload,
            headers=headers,
            timeout=120,
        )
        return r.json()["name"]
