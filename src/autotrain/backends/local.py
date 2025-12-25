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
        
        
        if isinstance(self.params, AutomaticSpeechRecognitionParams):
            
            config_path = f"{self.params.project_name}/training_config.json"
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            
            config_dict = self.params.dict()
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
            
            
            WORKSPACE_ROOT = os.path.abspath(".")  
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONPATH"] = os.pathsep.join(sys.path)
            command = f"{sys.executable} -m autotrain.trainers.automatic_speech_recognition.__main__ --training_config \"{config_path}\""

            print(f"[DEBUG] Subprocess command: {command}")
            logger.info(f"Running ASR command: {command}")
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"Config path: {config_path}, exists: {os.path.exists(config_path)}")
            logger.info(f"Python executable: {sys.executable}")
            logger.info(f"Environment PATH: {os.environ.get('PATH')}")
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=open("asr.log", "w", encoding="utf-8"),
                stderr=subprocess.STDOUT,  
                env=env,
                cwd=WORKSPACE_ROOT
            )
            
            
            from autotrain.app.db import AutoTrainDB
            DB = AutoTrainDB("autotrain.db")
            DB.add_job(process.pid)
            
            self.job_id = str(process.pid)
            logger.info(f"ASR Training started with PID: {self.job_id}")

           
                
            return
            
        
        params_json = self.env_vars["PARAMS"]
        task_id = int(self.env_vars["TASK_ID"])
        training_pid = run_training(params, task_id, local=True, wait=self.wait)
        if not self.wait:
            logger.info(f"Training PID: {training_pid}")
        return training_pid
