import enum
from dataclasses import dataclass

from autotrain.languages import SUPPORTED_LANGUAGES
from autotrain.tasks import TASKS


class LearningRate:
    TYPE = "float"
    MIN_VALUE = 1e-7
    MAX_VALUE = 1e-1
    DEFAULT = 1e-3
    STREAMLIT_INPUT = "number_input"
    PRETTY_NAME = "Learning Rate"


class Optimizer:
    TYPE = "str"
    DEFAULT = "adam"
    CHOICES = ["adam", "sgd"]
    STREAMLIT_INPUT = "selectbox"
    PRETTY_NAME = "Optimizer"


class Scheduler:
    TYPE = "str"
    DEFAULT = "linear"
    CHOICES = ["linear", "cosine"]
    STREAMLIT_INPUT = "selectbox"
    PRETTY_NAME = "Scheduler"


class BatchSize:
    TYPE = "int"
    MIN_VALUE = 1
    MAX_VALUE = 128
    DEFAULT = 8
    STREAMLIT_INPUT = "number_input"
    PRETTY_NAME = "Batch Size"


class Epochs:
    TYPE = "int"
    MIN_VALUE = 1
    MAX_VALUE = 1000
    DEFAULT = 10
    STREAMLIT_INPUT = "number_input"
    PRETTY_NAME = "Epochs"


class PercentageWarmup:
    TYPE = "float"
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0
    DEFAULT = 0.1
    STREAMLIT_INPUT = "number_input"
    PRETTY_NAME = "Percentage Warmup"


class GradientAccumulationSteps:
    TYPE = "int"
    MIN_VALUE = 1
    MAX_VALUE = 100
    DEFAULT = 1
    STREAMLIT_INPUT = "number_input"
    PRETTY_NAME = "Gradient Accumulation Steps"


class WeightDecay:
    TYPE = "float"
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0
    DEFAULT = 0.0
    STREAMLIT_INPUT = "number_input"
    PRETTY_NAME = "Weight Decay"


class SourceLanguage:
    TYPE = "str"
    DEFAULT = "en"
    CHOICES = SUPPORTED_LANGUAGES
    STREAMLIT_INPUT = "selectbox"
    PRETTY_NAME = "Source Language"


class TargetLanguage:
    TYPE = "str"
    DEFAULT = "en"
    CHOICES = SUPPORTED_LANGUAGES
    STREAMLIT_INPUT = "selectbox"
    PRETTY_NAME = "Target Language"


class NumModels:
    TYPE = "int"
    MIN_VALUE = 1
    MAX_VALUE = 25
    DEFAULT = 5
    STREAMLIT_INPUT = "number_input"
    PRETTY_NAME = "Number of Models"


@dataclass
class Params:
    task: str
    training_type: str

    def __post_init__(self):
        # task should be one of the keys in TASKS
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS.keys()}")
        self.task_id = TASKS[self.task]

        if self.training_type not in ("autotrain", "hub_model"):
            raise ValueError("training_type must be either autotrain or hub_model")

    def _text_binary_classification(self):
        if self.training_type == "hub_model":
            return {
                "learning_rate": LearningRate,
                "optimizer": Optimizer,
                "scheduler": Scheduler,
                "batch_size": BatchSize,
                "epochs": Epochs,
                "percentage_warmup": PercentageWarmup,
                "gradient_accumulation_steps": GradientAccumulationSteps,
                "weight_decay": WeightDecay,
            }
        return {
            "source_language": SourceLanguage,
            "num_models": NumModels,
        }

    def _text_multi_class_classification(self):
        return self._text_binary_classification()

    def get(self):
        if self.task == "text_binary_classification":
            return self._text_binary_classification()

        if self.task == "text_multi_class_classification":
            return self._text_multi_class_classification()
