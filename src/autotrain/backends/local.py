from autotrain import logger
from autotrain.app_utils import run_training
from autotrain.backends.base import BaseBackend


class LocalRunner(BaseBackend):
    def create(self):
        logger.info("Starting local training...")
        params = self.env_vars["PARAMS"]
        task_id = int(self.env_vars["TASK_ID"])
        wait = self.backend == "local-cli"
        training_pid = run_training(params, task_id, local=True, wait=wait)
        if not self.wait:
            logger.info(f"Training PID: {training_pid}")
        return training_pid
