from autotrain import logger
from autotrain.backends.base import BaseBackend


class NVCFRunner(BaseBackend)
    job_name: str
    env_vars: dict
    backend: str

    def __post_init__(self):
        self.nvcf_api = os.environ.get("NVCF_API")
        self.hf_token = self.env_vars["HF_TOKEN"]
        self.instance_map = {
            "nvcf-l40sx1": {"id": "67bb8939-c932-429a-a446-8ae898311856"},
            "nvcf-h100x1": {"id": "848348f8-a4e2-4242-bce9-6baa1bd70a66"},
            "nvcf-h100x2": {"id": "fb006a89-451e-4d9c-82b5-33eff257e0bf"},
            "nvcf-h100x4": {"id": "21bae5af-87e5-4132-8fc0-bf3084e59a57"},
            "nvcf-h100x8": {"id": "6e0c2af6-5368-47e0-b15e-c070c2c92018"},
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

    def _conf_nvcf(self, token, nvcf_type, url, method="POST", payload=None):
        logger.info(f"{self.job_name}: {method} - Configuring NVCF {nvcf_type}.")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        try:
            if method.upper() == "POST":
                response = requests.post(url, headers=headers, json=payload, timeout=30)
            else:

                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()

            if response.status_code == 202:
                logger.info(
                    f"{self.job_name}: {method} - Successfully submitted NVCF job. Polling reqId for completion"
                )
                response_data = response.json()
                nvcf_reqid = response_data.get("nvcfRequestId")
                if nvcf_reqid:
                    logger.info(f"{self.job_name}: nvcfRequestId: {nvcf_reqid}")
                    return nvcf_reqid
                logger.warning(f"{self.job_name}: nvcfRequestId key is missing in the response body")
                return None

            result = response.json()
            result_obj = self._convert_dict_to_object(result)
            logger.info(f"{self.job_name}: {method} - Successfully processed NVCF {nvcf_type}.")
            return result_obj

        except requests.HTTPError as http_err:
            # Log the response body for more context
            error_message = http_err.response.text if http_err.response else "No additional error information."
            logger.error(
                f"{self.job_name}: HTTP error occurred processing NVCF {nvcf_type} with {method} request: {http_err}. "
                f"Error details: {error_message}"
            )
            raise Exception(f"HTTP Error {http_err.response.status_code}: {http_err}. Details: {error_message}")

        except (requests.Timeout, ConnectionError) as err:
            logger.error(f"{self.job_name}: Failed to process NVCF {nvcf_type} with {method} request - {repr(err)}")
            raise Exception(f"Unreachable, please try again later: {err}")

    def _poll_nvcf(self, url, token, method="get", timeout=86400, interval=30, op="poll"):
        timeout = float(timeout)
        interval = float(interval)
        start_time = time.time()
        success = False
        last_full_log = ""

        while time.time() - start_time < timeout:
            try:
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
                if method.upper() == "GET":
                    response = requests.get(url, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                if response.status_code == 404 and success:
                    break

                response.raise_for_status()

                try:
                    data = response.json()
                except ValueError:
                    logger.error("Failed to parse JSON from response")
                    continue

                if response.status_code == 500:
                    logger.error("Training failed")
                    if "detail" in data:
                        detail_message = data["detail"]
                        for line in detail_message.split("\n"):
                            if line.strip():
                                print(line)
                    break

                if response.status_code in [200, 202]:
                    logger.info(
                        f"{self.job_name}: {method} - {response.status_code} - {'Polling completed' if response.status_code == 200 else 'Polling reqId for completion'}"
                    )

                    if "log" in data:
                        current_full_log = data["log"]
                        if current_full_log != last_full_log:
                            new_log_content = current_full_log[len(last_full_log) :]
                            for line in new_log_content.split("\n"):
                                if line.strip():
                                    print(line)
                            last_full_log = current_full_log

                    if response.status_code == 200:
                        success = True

            except requests.HTTPError as http_err:
                if not (http_err.response.status_code == 404 and success):
                    logger.error(f"HTTP error occurred: {http_err}")
            except (requests.ConnectionError, ValueError) as err:
                logger.error(f"Error while handling request: {err}")

            time.sleep(interval)

        if not success:
            raise TimeoutError(f"Operation '{op}' did not complete successfully within the timeout period.")

    def create(self):
        nvcf_url_submit = f"{self.nvcf_api}/invoke/{self.instance_map[self.backend]['id']}"
        org_name = os.environ.get("SPACE_ID")
        if org_name is None:
            raise ValueError("SPACE_ID environment variable is not set")
        org_name = org_name.split("/")[0]
        nvcf_fr_payload = {
            "cmd": [
                "conda",
                "run",
                "--no-capture-output",
                "-p",
                "/app/env",
                "python",
                "-u",
                "-m",
                "uvicorn",
                "autotrain.api:api",
                "--host",
                "0.0.0.0",
                "--port",
                "7860",
            ],
            "env": {key: value for key, value in self.env_vars.items()},
            "ORG_NAME": org_name,
        }

        nvcf_fn_req = self._conf_nvcf(
            token=self.hf_token, nvcf_type="job_submit", url=nvcf_url_submit, method="POST", payload=nvcf_fr_payload
        )

        nvcf_url_reqpoll = f"{self.nvcf_api}/status/{nvcf_fn_req}"
        logger.info(f"{self.job_name}: Polling : {nvcf_url_reqpoll}")
        poll_thread = threading.Thread(
            target=self._poll_nvcf,
            kwargs={
                "url": nvcf_url_reqpoll,
                "token": self.hf_token,
                "method": "GET",
                "timeout": 172800,
                "interval": 20,
            },
        )
        poll_thread.start()
