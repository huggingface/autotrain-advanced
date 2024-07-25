from typing import Optional

from pydantic import Field

from autotrain.trainers.common import AutoTrainParams


class VLMTrainingParams(AutoTrainParams):
    model: str = Field("google/paligemma-3b-pt-224", title="Model name")
    project_name: str = Field("project-name", title="Output directory")

    # data params
    data_path: str = Field("data", title="Data path")
    train_split: str = Field("train", title="Train data config")
    valid_split: Optional[str] = Field(None, title="Validation data config")

    # trainer params
    trainer: str = Field("vqa", title="Trainer type")  # captioning, vqa, segmentation, detection
    log: str = Field("none", title="Logging using experiment tracking")
    disable_gradient_checkpointing: bool = Field(False, title="Gradient checkpointing")
    logging_steps: int = Field(-1, title="Logging steps")
    eval_strategy: str = Field("epoch", title="Evaluation strategy")
    save_total_limit: int = Field(1, title="Save total limit")
    auto_find_batch_size: bool = Field(False, title="Auto find batch size")
    mixed_precision: Optional[str] = Field(None, title="fp16, bf16, or None")
    lr: float = Field(3e-5, title="Learning rate")
    epochs: int = Field(1, title="Number of training epochs")
    batch_size: int = Field(2, title="Training batch size")
    warmup_ratio: float = Field(0.1, title="Warmup proportion")
    gradient_accumulation: int = Field(4, title="Gradient accumulation steps")
    optimizer: str = Field("adamw_torch", title="Optimizer")
    scheduler: str = Field("linear", title="Scheduler")
    weight_decay: float = Field(0.0, title="Weight decay")
    max_grad_norm: float = Field(1.0, title="Max gradient norm")
    seed: int = Field(42, title="Seed")

    # peft
    quantization: Optional[str] = Field("int4", title="int4, int8, or None")
    target_modules: Optional[str] = Field("all-linear", title="Target modules")
    merge_adapter: bool = Field(False, title="Merge adapter")
    peft: bool = Field(False, title="Use PEFT")
    lora_r: int = Field(16, title="Lora r")
    lora_alpha: int = Field(32, title="Lora alpha")
    lora_dropout: float = Field(0.05, title="Lora dropout")

    # column mappings
    image_column: Optional[str] = Field("image", title="Image column")
    text_column: str = Field("text", title="Text (answer) column")
    prompt_text_column: Optional[str] = Field("prompt", title="Prompt (prefix) column")

    # push to hub
    push_to_hub: bool = Field(False, title="Push to hub")
    username: Optional[str] = Field(None, title="Hugging Face Username")
    token: Optional[str] = Field(None, title="Huggingface token")
