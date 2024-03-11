
import os
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from accelerate import PartialState

REQUEST_ID = os.getenv("NVCF-REQID")
LOG_PAYLOAD = """
    {{
        \\"id\\": \\"{request_id}\\",
        \\"progress\\": \\"{progress}\\"
    }}
"""

LOG_PATH = "/var/inf/response/{request_id}/log.partialprogress"

class NVCFCallback(TrainerCallback):
    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        
        if REQUEST_ID and PartialState().is_main_process:
            path = LOG_PATH.format(request_id = REQUEST_ID)
            progress = int(state.global_step / state.max_steps)
            payload = LOG_PAYLOAD.format(
                request_id = REQUEST_ID,
                progress = progress
            )
            with open(path, mode = "w") as log_file:
                log_file.write(payload)


