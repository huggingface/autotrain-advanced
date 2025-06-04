from typing import Optional

from pydantic import Field

from autotrain.trainers.common import AutoTrainParams


class VLMTrainingParams(AutoTrainParams):
    """
    VLMTrainingParams

    Attributes:
        model (str): Model name. Default is "google/paligemma-3b-pt-224".
        project_name (str): Output directory. Default is "project-name".

        data_path (str): Data path. Default is "data".
        train_split (str): Train data config. Default is "train".
        valid_split (Optional[str]): Validation data config. Default is None.

        trainer (str): Trainer type (captioning, vqa, segmentation, detection). Default is "vqa".
        log (str): Logging using experiment tracking. Default is "none".
        disable_gradient_checkpointing (bool): Gradient checkpointing. Default is False.
        logging_steps (int): Logging steps. Default is -1.
        eval_strategy (str): Evaluation strategy. Default is "epoch".
        save_total_limit (int): Save total limit. Default is 1.
        auto_find_batch_size (bool): Auto find batch size. Default is False.
        mixed_precision (Optional[str]): Mixed precision (fp16, bf16, or None). Default is None.
        lr (float): Learning rate. Default is 3e-5.
        epochs (int): Number of training epochs. Default is 1.
        batch_size (int): Training batch size. Default is 2.
        warmup_ratio (float): Warmup proportion. Default is 0.1.
        gradient_accumulation (int): Gradient accumulation steps. Default is 4.
        optimizer (str): Optimizer. Default is "adamw_torch".
        scheduler (str): Scheduler. Default is "linear".
        weight_decay (float): Weight decay. Default is 0.0.
        max_grad_norm (float): Max gradient norm. Default is 1.0.
        seed (int): Seed. Default is 42.

        quantization (Optional[str]): Quantization (int4, int8, or None). Default is "int4".
        target_modules (Optional[str]): Target modules. Default is "all-linear".
        merge_adapter (bool): Merge adapter. Default is False.
        peft (bool): Use PEFT. Default is False.
        lora_r (int): Lora r. Default is 16.
        lora_alpha (int): Lora alpha. Default is 32.
        lora_dropout (float): Lora dropout. Default is 0.05.

        image_column (Optional[str]): Image column. Default is "image".
        text_column (str): Text (answer) column. Default is "text".
        prompt_text_column (Optional[str]): Prompt (prefix) column. Default is "prompt".

        push_to_hub (bool): Push to hub. Default is False.
        username (Optional[str]): Hugging Face Username. Default is None.
        token (Optional[str]): Huggingface token. Default is None.
    """

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
