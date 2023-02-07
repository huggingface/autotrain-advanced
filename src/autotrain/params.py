import enum
from dataclasses import dataclass

from autotrain.tasks import TASKS


class LearningRate(enum.Enum):
    TYPE = "float"
    MIN_VALUE = 1e-7
    MAX_VALUE = 1e-1
    DEFAULT = 1e-3
    STREAMLIT_INPUT = "number_input"


class Optimizer(enum.Enum):
    TYPE = "str"
    DEFAULT = "adam"
    CHOICES = ["adam", "sgd"]
    STREAMLIT_INPUT = "selectbox"


class Scheduler(enum.Enum):
    TYPE = "str"
    DEFAULT = "linear"
    CHOICES = ["linear", "cosine"]
    STREAMLIT_INPUT = "selectbox"


class BatchSize(enum.Enum):
    TYPE = "int"
    MIN_VALUE = 1
    MAX_VALUE = 128
    DEFAULT = 8
    STREAMLIT_INPUT = "number_input"


class Epochs(enum.Enum):
    TYPE = "int"
    MIN_VALUE = 1
    MAX_VALUE = 1000
    DEFAULT = 10
    STREAMLIT_INPUT = "number_input"


class PercentageWarmup(enum.Enum):
    TYPE = "float"
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0
    DEFAULT = 0.1
    STREAMLIT_INPUT = "number_input"


class GradientAccumulationSteps(enum.Enum):
    TYPE = "int"
    MIN_VALUE = 1
    MAX_VALUE = 100
    DEFAULT = 1
    STREAMLIT_INPUT = "number_input"


@dataclass
class Params:
    task: str

    def __post_init__(self):
        # task should be one of the keys in TASKS
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS.keys()}")
        self.task_id = TASKS[self.task]

    def _nlp_binary_classification(self):
        return {
            "learning_rate": LearningRate,
            "optimizer": Optimizer,
            "scheduler": Scheduler,
            "batch_size": BatchSize,
            "epochs": Epochs,
            "percentage_warmup": PercentageWarmup,
            "gradient_accumulation_steps": GradientAccumulationSteps,
        }

    def _nlp_multi_class_classification(self):
        return self._nlp_binary_classification()

    def get(self):
        if self.task == "binary_classification":
            return self._nlp_binary_classification()

        if self.task == "multi_class_classification":
            return self._nlp_multi_class_classification()
