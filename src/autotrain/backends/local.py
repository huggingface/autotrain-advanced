from autotrain import logger
from autotrain.backends.base import BaseBackend
from autotrain.utils import run_training


class LocalRunner(BaseBackend):
    """
    LocalRunner is a class that inherits from BaseBackend and is responsible for managing local training tasks.

    Methods:
        create():
            Starts the local training process by retrieving parameters and task ID from environment variables.
            Logs the start of the training process.
            Runs the training with the specified parameters and task ID.
            If the `wait` attribute is False, logs the training process ID (PID).
            Returns the training process ID (PID).
    """

    def create(self):
        logger.info("Starting local training...")
        params = self.env_vars["PARAMS"]
        task_id = int(self.env_vars["TASK_ID"])
        training_pid = run_training(params, task_id, local=True, wait=self.wait)
        if not self.wait:
            logger.info(f"Training PID: {training_pid}")
        return training_pid
