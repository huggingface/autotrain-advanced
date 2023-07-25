from dataclasses import dataclass
from typing import Literal

import gradio as gr
from pydantic import BaseModel, Field

from autotrain.languages import SUPPORTED_LANGUAGES
from autotrain.tasks import TASKS


class DreamBoothTrainingParams(BaseModel):
    model_name: str = Field(None, title="Model name")
    revision: str = Field(None, title="Revision")
    tokenizer: str = Field(None, title="Tokenizer, if different from model")
    image_path: str = Field(None, title="Image path")
    class_image_path: str = Field(None, title="Class image path")
    instance_prompt: str = Field(None, title="Instance prompt")
    class_prompt: str = Field(None, title="Class prompt")
    validation_prompt: str = Field(None, title="Validation prompt")
    num_validation_images: int = Field(4, title="Number of validation images")
    validation_epochs: int = Field(50, title="Validation epochs")
    with_prior_preservation: bool = Field(False, title="With prior preservation")
    prior_loss_weight: float = Field(1.0, title="Prior loss weight")
    num_class_images: int = Field(100, title="Number of class images")
    output_dir: str = Field("lora-dreambooth-model", title="Output directory")
    seed: int = Field(42, title="Seed")
    resolution: int = Field(512, title="Resolution")
    center_crop: bool = Field(False, title="Center crop")
    train_text_encoder: bool = Field(False, title="Train text encoder")
    train_batch_size: int = Field(4, title="Train batch size")
    sample_batch_size: int = Field(4, title="Sample batch size")
    num_train_epochs: int = Field(1, title="Number of training epochs")
    max_train_steps: int = Field(None, title="Max train steps")
    checkpointing_steps: int = Field(500, title="Checkpointing steps")
    checkpoints_total_limit: int = Field(None, title="Checkpoints total limit")
    resume_from_checkpoint: str = Field(None, title="Resume from checkpoint")
    gradient_accumulation_steps: int = Field(1, title="Gradient accumulation steps")
    gradient_checkpointing: bool = Field(False, title="Gradient checkpointing")
    learning_rate: float = Field(5e-4, title="Learning rate")
    scale_lr: bool = Field(False, title="Scale learning rate")
    lr_scheduler: str = Field("constant", title="Learning rate scheduler")
    lr_warmup_steps: int = Field(0, title="Learning rate warmup steps")
    lr_num_cycles: int = Field(1, title="Learning rate num cycles")
    lr_power: float = Field(1.0, title="Learning rate power")
    dataloader_num_workers: int = Field(0, title="Dataloader num workers")
    use_8bit_adam: bool = Field(False, title="Use 8bit adam")
    adam_beta1: float = Field(0.9, title="Adam beta 1")
    adam_beta2: float = Field(0.999, title="Adam beta 2")
    adam_weight_decay: float = Field(1e-2, title="Adam weight decay")
    adam_epsilon: float = Field(1e-8, title="Adam epsilon")
    max_grad_norm: float = Field(1.0, title="Max grad norm")
    push_to_hub: bool = Field(False, title="Push to hub")
    hub_token: str = Field(None, title="Hub token")
    hub_model_id: str = Field(None, title="Hub model id")
    logging_dir: str = Field("logs", title="Logging directory")
    allow_tf32: bool = Field(False, title="Allow TF32")
    report_to: str = Field("tensorboard", title="Report to")
    mixed_precision: str = Field(None, title="Mixed precision")
    prior_generation_precision: str = Field(None, title="Prior generation precision")
    local_rank: int = Field(-1, title="Local rank")
    enable_xformers_memory_efficient_attention: bool = Field(False, title="Enable xformers memory efficient attention")
    pre_compute_text_embeddings: bool = Field(False, title="Pre compute text embeddings")
    tokenizer_max_length: int = Field(None, title="Tokenizer max length")
    text_encoder_use_attention_mask: bool = Field(False, title="Text encoder use attention mask")
    validation_images: str = Field(None, title="Validation images")
    class_labels_conditioning: str = Field(None, title="Class labels conditioning")
    rank: int = Field(4, title="Rank")
    xl: bool = Field(False, title="XL")


class LoraR:
    TYPE = "int"
    MIN_VALUE = 1
    MAX_VALUE = 100
    DEFAULT = 16
    STEP = 1
    STREAMLIT_INPUT = "number_input"
    PRETTY_NAME = "LoRA R"
    GRADIO_INPUT = gr.Slider(minimum=MIN_VALUE, maximum=MAX_VALUE, value=DEFAULT, step=STEP)


class LoraAlpha:
    TYPE = "int"
    MIN_VALUE = 1
    MAX_VALUE = 256
    DEFAULT = 32
    STEP = 1
    STREAMLIT_INPUT = "number_input"
    PRETTY_NAME = "LoRA Alpha"
    GRADIO_INPUT = gr.Slider(minimum=MIN_VALUE, maximum=MAX_VALUE, value=DEFAULT, step=STEP)


class LoraDropout:
    TYPE = "float"
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0
    DEFAULT = 0.05
    STEP = 0.01
    STREAMLIT_INPUT = "number_input"
    PRETTY_NAME = "LoRA Dropout"
    GRADIO_INPUT = gr.Slider(minimum=MIN_VALUE, maximum=MAX_VALUE, value=DEFAULT, step=STEP)


class LearningRate:
    TYPE = "float"
    MIN_VALUE = 1e-7
    MAX_VALUE = 1e-1
    DEFAULT = 1e-3
    STEP = 1e-6
    FORMAT = "%.2E"
    STREAMLIT_INPUT = "number_input"
    PRETTY_NAME = "Learning Rate"
    GRADIO_INPUT = gr.Slider(minimum=MIN_VALUE, maximum=MAX_VALUE, value=DEFAULT, step=STEP)


class LMLearningRate(LearningRate):
    DEFAULT = 5e-5


class Optimizer:
    TYPE = "str"
    DEFAULT = "adamw_torch"
    CHOICES = ["adamw_torch", "adamw_hf", "sgd", "adafactor", "adagrad"]
    STREAMLIT_INPUT = "selectbox"
    PRETTY_NAME = "Optimizer"
    GRADIO_INPUT = gr.Dropdown(CHOICES, value=DEFAULT)


class LMTrainingType:
    TYPE = "str"
    DEFAULT = "generic"
    CHOICES = ["generic", "chat"]
    STREAMLIT_INPUT = "selectbox"
    PRETTY_NAME = "LM Training Type"
    GRAIDO_INPUT = gr.Dropdown(CHOICES, value=DEFAULT)


class Scheduler:
    TYPE = "str"
    DEFAULT = "linear"
    CHOICES = ["linear", "cosine"]
    STREAMLIT_INPUT = "selectbox"
    PRETTY_NAME = "Scheduler"
    GRADIO_INPUT = gr.Dropdown(CHOICES, value=DEFAULT)


class TrainBatchSize:
    TYPE = "int"
    MIN_VALUE = 1
    MAX_VALUE = 128
    DEFAULT = 2
    STEP = 2
    STREAMLIT_INPUT = "number_input"
    PRETTY_NAME = "Train Batch Size"
    GRADIO_INPUT = gr.Slider(minimum=MIN_VALUE, maximum=MAX_VALUE, value=DEFAULT, step=STEP)


class LMTrainBatchSize(TrainBatchSize):
    DEFAULT = 4


class Epochs:
    TYPE = "int"
    MIN_VALUE = 1
    MAX_VALUE = 1000
    DEFAULT = 10
    STREAMLIT_INPUT = "number_input"
    PRETTY_NAME = "Epochs"
    GRADIO_INPUT = gr.Number(value=DEFAULT)


class LMEpochs(Epochs):
    DEFAULT = 1


class PercentageWarmup:
    TYPE = "float"
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0
    DEFAULT = 0.1
    STEP = 0.01
    STREAMLIT_INPUT = "number_input"
    PRETTY_NAME = "Percentage Warmup"
    GRADIO_INPUT = gr.Slider(minimum=MIN_VALUE, maximum=MAX_VALUE, value=DEFAULT, step=STEP)


class GradientAccumulationSteps:
    TYPE = "int"
    MIN_VALUE = 1
    MAX_VALUE = 100
    DEFAULT = 1
    STREAMLIT_INPUT = "number_input"
    PRETTY_NAME = "Gradient Accumulation Steps"
    GRADIO_INPUT = gr.Number(value=DEFAULT)


class WeightDecay:
    TYPE = "float"
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0
    DEFAULT = 0.0
    STREAMLIT_INPUT = "number_input"
    PRETTY_NAME = "Weight Decay"
    GRADIO_INPUT = gr.Number(value=DEFAULT)


class SourceLanguage:
    TYPE = "str"
    DEFAULT = "en"
    CHOICES = SUPPORTED_LANGUAGES
    STREAMLIT_INPUT = "selectbox"
    PRETTY_NAME = "Source Language"
    GRADIO_INPUT = gr.Dropdown(CHOICES, value=DEFAULT)


class TargetLanguage:
    TYPE = "str"
    DEFAULT = "en"
    CHOICES = SUPPORTED_LANGUAGES
    STREAMLIT_INPUT = "selectbox"
    PRETTY_NAME = "Target Language"
    GRADIO_INPUT = gr.Dropdown(CHOICES, value=DEFAULT)


class NumModels:
    TYPE = "int"
    MIN_VALUE = 1
    MAX_VALUE = 25
    DEFAULT = 1
    STREAMLIT_INPUT = "number_input"
    PRETTY_NAME = "Number of Models"
    GRADIO_INPUT = gr.Slider(minimum=MIN_VALUE, maximum=MAX_VALUE, value=DEFAULT, step=1)


class DBNumSteps:
    TYPE = "int"
    MIN_VALUE = 100
    MAX_VALUE = 10000
    DEFAULT = 1500
    STREAMLIT_INPUT = "number_input"
    PRETTY_NAME = "Number of Steps"
    GRADIO_INPUT = gr.Slider(minimum=MIN_VALUE, maximum=MAX_VALUE, value=DEFAULT, step=100)


class DBTextEncoderStepsPercentage:
    TYPE = "int"
    MIN_VALUE = 1
    MAX_VALUE = 100
    DEFAULT = 30
    STREAMLIT_INPUT = "number_input"
    PRETTY_NAME = "Text encoder steps percentage"
    GRADIO_INPUT = gr.Slider(minimum=MIN_VALUE, maximum=MAX_VALUE, value=DEFAULT, step=1)


class DBPriorPreservation:
    TYPE = "bool"
    DEFAULT = False
    STREAMLIT_INPUT = "checkbox"
    PRETTY_NAME = "Prior preservation"
    GRADIO_INPUT = gr.Dropdown(["True", "False"], value="False")


class ImageSize:
    TYPE = "int"
    MIN_VALUE = 64
    MAX_VALUE = 2048
    DEFAULT = 512
    STREAMLIT_INPUT = "number_input"
    PRETTY_NAME = "Image Size"
    GRADIO_INPUT = gr.Slider(minimum=MIN_VALUE, maximum=MAX_VALUE, value=DEFAULT, step=64)


class DreamboothConceptType:
    TYPE = "str"
    DEFAULT = "person"
    CHOICES = ["person", "object"]
    STREAMLIT_INPUT = "selectbox"
    PRETTY_NAME = "Concept Type"
    GRADIO_INPUT = gr.Dropdown(CHOICES, value=DEFAULT)


class SourceLanguageUnk:
    TYPE = "str"
    DEFAULT = "unk"
    CHOICES = ["unk"]
    STREAMLIT_INPUT = "selectbox"
    PRETTY_NAME = "Source Language"
    GRADIO_INPUT = gr.Dropdown(CHOICES, value=DEFAULT)


class HubModel:
    TYPE = "str"
    DEFAULT = "bert-base-uncased"
    PRETTY_NAME = "Hub Model"
    GRADIO_INPUT = gr.Textbox(lines=1, max_lines=1, label="Hub Model")


class TextBinaryClassificationParams(BaseModel):
    task: Literal["text_binary_classification"]
    learning_rate: float = Field(5e-5, title="Learning rate")
    num_train_epochs: int = Field(3, title="Number of training epochs")
    max_seq_length: int = Field(128, title="Max sequence length")
    train_batch_size: int = Field(32, title="Training batch size")
    warmup_ratio: float = Field(0.1, title="Warmup proportion")
    gradient_accumulation_steps: int = Field(1, title="Gradient accumulation steps")
    optimizer: str = Field("adamw_torch", title="Optimizer")
    scheduler: str = Field("linear", title="Scheduler")
    weight_decay: float = Field(0.0, title="Weight decay")
    max_grad_norm: float = Field(1.0, title="Max gradient norm")
    seed: int = Field(42, title="Seed")


class TextMultiClassClassificationParams(BaseModel):
    task: Literal["text_multi_class_classification"]
    learning_rate: float = Field(5e-5, title="Learning rate")
    num_train_epochs: int = Field(3, title="Number of training epochs")
    max_seq_length: int = Field(128, title="Max sequence length")
    train_batch_size: int = Field(32, title="Training batch size")
    warmup_ratio: float = Field(0.1, title="Warmup proportion")
    gradient_accumulation_steps: int = Field(1, title="Gradient accumulation steps")
    optimizer: str = Field("adamw_torch", title="Optimizer")
    scheduler: str = Field("linear", title="Scheduler")
    weight_decay: float = Field(0.0, title="Weight decay")
    max_grad_norm: float = Field(1.0, title="Max gradient norm")
    seed: int = Field(42, title="Seed")


class DreamboothParams(BaseModel):
    task: Literal["dreambooth"]
    num_steps: int = Field(1500, title="Number of steps")
    image_size: int = Field(512, title="Image size")
    text_encoder_steps_percentage: int = Field(30, title="Text encoder steps percentage")
    prior_preservation: bool = Field(False, title="Prior preservation")
    learning_rate: float = Field(2e-6, title="Learning rate")
    train_batch_size: int = Field(1, title="Training batch size")


class ImageBinaryClassificationParams(BaseModel):
    task: Literal["image_binary_classification"]
    learning_rate: float = Field(3e-5, title="Learning rate")
    num_train_epochs: int = Field(3, title="Number of training epochs")
    train_batch_size: int = Field(8, title="Training batch size")
    warmup_ratio: float = Field(0.1, title="Warmup proportion")
    gradient_accumulation_steps: int = Field(1, title="Gradient accumulation steps")
    optimizer: str = Field("adamw_torch", title="Optimizer")
    scheduler: str = Field("linear", title="Scheduler")
    weight_decay: float = Field(0.0, title="Weight decay")
    max_grad_norm: float = Field(1.0, title="Max gradient norm")
    seed: int = Field(42, title="Seed")


class ImageMultiClassClassificationParams(BaseModel):
    task: Literal["image_multi_class_classification"]
    learning_rate: float = Field(3e-5, title="Learning rate")
    num_train_epochs: int = Field(3, title="Number of training epochs")
    train_batch_size: int = Field(8, title="Training batch size")
    warmup_ratio: float = Field(0.1, title="Warmup proportion")
    gradient_accumulation_steps: int = Field(1, title="Gradient accumulation steps")
    optimizer: str = Field("adamw_torch", title="Optimizer")
    scheduler: str = Field("linear", title="Scheduler")
    weight_decay: float = Field(0.0, title="Weight decay")
    max_grad_norm: float = Field(1.0, title="Max gradient norm")
    seed: int = Field(42, title="Seed")


class LMTrainingParams(BaseModel):
    task: Literal["lm_training"]
    learning_rate: float = Field(3e-5, title="Learning rate")
    num_train_epochs: int = Field(3, title="Number of training epochs")
    train_batch_size: int = Field(8, title="Training batch size")
    warmup_ratio: float = Field(0.1, title="Warmup proportion")
    gradient_accumulation_steps: int = Field(1, title="Gradient accumulation steps")
    optimizer: str = Field("adamw_torch", title="Optimizer")
    scheduler: str = Field("linear", title="Scheduler")
    weight_decay: float = Field(0.0, title="Weight decay")
    max_grad_norm: float = Field(1.0, title="Max gradient norm")
    seed: int = Field(42, title="Seed")
    add_eos_token: bool = Field(True, title="Add EOS token")
    block_size: int = Field(-1, title="Block size")
    lora_r: int = Field(16, title="Lora r")
    lora_alpha: int = Field(32, title="Lora alpha")
    lora_dropout: float = Field(0.05, title="Lora dropout")
    training_type: str = Field("generic", title="Training type")
    train_on_inputs: bool = Field(False, title="Train on inputs")


@dataclass
class Params:
    task: str
    param_choice: str
    model_choice: str

    def __post_init__(self):
        # task should be one of the keys in TASKS
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS.keys()}")
        self.task_id = TASKS[self.task]

        if self.param_choice not in ("autotrain", "manual"):
            raise ValueError("param_choice must be either autotrain or manual")

        if self.model_choice not in ("autotrain", "hub_model"):
            raise ValueError("model_choice must be either autotrain or hub_model")

    def _dreambooth(self):
        if self.param_choice == "manual":
            return {
                "hub_model": HubModel,
                "image_size": ImageSize,
                "learning_rate": LearningRate,
                "train_batch_size": TrainBatchSize,
                "num_steps": DBNumSteps,
                "text_encoder_steps_percentage": DBTextEncoderStepsPercentage,
                "prior_preservation": DBPriorPreservation,
            }
        if self.param_choice == "autotrain":
            if self.model_choice == "hub_model":
                return {
                    "hub_model": HubModel,
                    "image_size": ImageSize,
                    "num_models": NumModels,
                }
            else:
                return {
                    "num_models": NumModels,
                }

    def _tabular_binary_classification(self):
        return {
            "num_models": NumModels,
        }

    def _lm_training(self):
        if self.param_choice == "manual":
            return {
                "hub_model": HubModel,
                "learning_rate": LMLearningRate,
                "optimizer": Optimizer,
                "scheduler": Scheduler,
                "train_batch_size": LMTrainBatchSize,
                "num_train_epochs": LMEpochs,
                "percentage_warmup": PercentageWarmup,
                "gradient_accumulation_steps": GradientAccumulationSteps,
                "weight_decay": WeightDecay,
                "lora_r": LoraR,
                "lora_alpha": LoraAlpha,
                "lora_dropout": LoraDropout,
                "training_type": LMTrainingType,
            }
        if self.param_choice == "autotrain":
            if self.model_choice == "autotrain":
                return {
                    "num_models": NumModels,
                    "training_type": LMTrainingType,
                }
            else:
                return {
                    "hub_model": HubModel,
                    "num_models": NumModels,
                    "training_type": LMTrainingType,
                }
        raise ValueError("param_choice must be either autotrain or manual")

    def _tabular_multi_class_classification(self):
        return self._tabular_binary_classification()

    def _tabular_single_column_regression(self):
        return self._tabular_binary_classification()

    def tabular_multi_label_classification(self):
        return self._tabular_binary_classification()

    def _text_binary_classification(self):
        if self.param_choice == "manual":
            return {
                "hub_model": HubModel,
                "learning_rate": LearningRate,
                "optimizer": Optimizer,
                "scheduler": Scheduler,
                "train_batch_size": TrainBatchSize,
                "num_train_epochs": Epochs,
                "percentage_warmup": PercentageWarmup,
                "gradient_accumulation_steps": GradientAccumulationSteps,
                "weight_decay": WeightDecay,
            }
        if self.param_choice == "autotrain":
            if self.model_choice == "autotrain":
                return {
                    "source_language": SourceLanguage,
                    "num_models": NumModels,
                }
            return {
                "hub_model": HubModel,
                "source_language": SourceLanguageUnk,
                "num_models": NumModels,
            }
        raise ValueError("param_choice must be either autotrain or manual")

    def _text_multi_class_classification(self):
        return self._text_binary_classification()

    def _text_entity_extraction(self):
        return self._text_binary_classification()

    def _text_single_column_regression(self):
        return self._text_binary_classification()

    def _text_natural_language_inference(self):
        return self._text_binary_classification()

    def _image_binary_classification(self):
        if self.param_choice == "manual":
            return {
                "hub_model": HubModel,
                "learning_rate": LearningRate,
                "optimizer": Optimizer,
                "scheduler": Scheduler,
                "train_batch_size": TrainBatchSize,
                "num_train_epochs": Epochs,
                "percentage_warmup": PercentageWarmup,
                "gradient_accumulation_steps": GradientAccumulationSteps,
                "weight_decay": WeightDecay,
            }
        if self.param_choice == "autotrain":
            if self.model_choice == "autotrain":
                return {
                    "num_models": NumModels,
                }
            return {
                "hub_model": HubModel,
                "num_models": NumModels,
            }
        raise ValueError("param_choice must be either autotrain or manual")

    def _image_multi_class_classification(self):
        return self._image_binary_classification()

    def get(self):
        if self.task in ("text_binary_classification", "text_multi_class_classification"):
            return self._text_binary_classification()

        if self.task == "text_entity_extraction":
            return self._text_entity_extraction()

        if self.task == "text_single_column_regression":
            return self._text_single_column_regression()

        if self.task == "text_natural_language_inference":
            return self._text_natural_language_inference()

        if self.task == "tabular_binary_classification":
            return self._tabular_binary_classification()

        if self.task == "tabular_multi_class_classification":
            return self._tabular_multi_class_classification()

        if self.task == "tabular_single_column_regression":
            return self._tabular_single_column_regression()

        if self.task == "tabular_multi_label_classification":
            return self.tabular_multi_label_classification()

        if self.task in ("image_binary_classification", "image_multi_class_classification"):
            return self._image_binary_classification()

        if self.task == "dreambooth":
            return self._dreambooth()

        if self.task == "lm_training":
            return self._lm_training()

        raise ValueError(f"task {self.task} not supported")
