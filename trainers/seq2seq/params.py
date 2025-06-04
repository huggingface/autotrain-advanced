from typing import Optional

from pydantic import Field

from autotrain.trainers.common import AutoTrainParams


class Seq2SeqParams(AutoTrainParams):
    """
    Seq2SeqParams is a configuration class for sequence-to-sequence training parameters.

    Attributes:
        data_path (str): Path to the dataset.
        model (str): Name of the model to be used. Default is "google/flan-t5-base".
        username (Optional[str]): Hugging Face Username.
        seed (int): Random seed for reproducibility. Default is 42.
        train_split (str): Name of the training data split. Default is "train".
        valid_split (Optional[str]): Name of the validation data split.
        project_name (str): Name of the project or output directory. Default is "project-name".
        token (Optional[str]): Hub Token for authentication.
        push_to_hub (bool): Whether to push the model to the Hugging Face Hub. Default is False.
        text_column (str): Name of the text column in the dataset. Default is "text".
        target_column (str): Name of the target text column in the dataset. Default is "target".
        lr (float): Learning rate for training. Default is 5e-5.
        epochs (int): Number of training epochs. Default is 3.
        max_seq_length (int): Maximum sequence length for input text. Default is 128.
        max_target_length (int): Maximum sequence length for target text. Default is 128.
        batch_size (int): Training batch size. Default is 2.
        warmup_ratio (float): Proportion of warmup steps. Default is 0.1.
        gradient_accumulation (int): Number of gradient accumulation steps. Default is 1.
        optimizer (str): Optimizer to be used. Default is "adamw_torch".
        scheduler (str): Learning rate scheduler to be used. Default is "linear".
        weight_decay (float): Weight decay for the optimizer. Default is 0.0.
        max_grad_norm (float): Maximum gradient norm for clipping. Default is 1.0.
        logging_steps (int): Number of steps between logging. Default is -1 (disabled).
        eval_strategy (str): Evaluation strategy. Default is "epoch".
        auto_find_batch_size (bool): Whether to automatically find the batch size. Default is False.
        mixed_precision (Optional[str]): Mixed precision training mode (fp16, bf16, or None).
        save_total_limit (int): Maximum number of checkpoints to save. Default is 1.
        peft (bool): Whether to use Parameter-Efficient Fine-Tuning (PEFT). Default is False.
        quantization (Optional[str]): Quantization mode (int4, int8, or None). Default is "int8".
        lora_r (int): LoRA-R parameter for PEFT. Default is 16.
        lora_alpha (int): LoRA-Alpha parameter for PEFT. Default is 32.
        lora_dropout (float): LoRA-Dropout parameter for PEFT. Default is 0.05.
        target_modules (str): Target modules for PEFT. Default is "all-linear".
        log (str): Logging method for experiment tracking. Default is "none".
        early_stopping_patience (int): Patience for early stopping. Default is 5.
        early_stopping_threshold (float): Threshold for early stopping. Default is 0.01.
    """

    data_path: str = Field(None, title="Data path")
    model: str = Field("google/flan-t5-base", title="Model name")
    username: Optional[str] = Field(None, title="Hugging Face Username")
    seed: int = Field(42, title="Seed")
    train_split: str = Field("train", title="Train split")
    valid_split: Optional[str] = Field(None, title="Validation split")
    project_name: str = Field("project-name", title="Output directory")
    token: Optional[str] = Field(None, title="Hub Token")
    push_to_hub: bool = Field(False, title="Push to hub")
    text_column: str = Field("text", title="Text column")
    target_column: str = Field("target", title="Target text column")
    lr: float = Field(5e-5, title="Learning rate")
    epochs: int = Field(3, title="Number of training epochs")
    max_seq_length: int = Field(128, title="Max sequence length")
    max_target_length: int = Field(128, title="Max target sequence length")
    batch_size: int = Field(2, title="Training batch size")
    warmup_ratio: float = Field(0.1, title="Warmup proportion")
    gradient_accumulation: int = Field(1, title="Gradient accumulation steps")
    optimizer: str = Field("adamw_torch", title="Optimizer")
    scheduler: str = Field("linear", title="Scheduler")
    weight_decay: float = Field(0.0, title="Weight decay")
    max_grad_norm: float = Field(1.0, title="Max gradient norm")
    logging_steps: int = Field(-1, title="Logging steps")
    eval_strategy: str = Field("epoch", title="Evaluation strategy")
    auto_find_batch_size: bool = Field(False, title="Auto find batch size")
    mixed_precision: Optional[str] = Field(None, title="fp16, bf16, or None")
    save_total_limit: int = Field(1, title="Save total limit")
    token: Optional[str] = Field(None, title="Hub Token")
    push_to_hub: bool = Field(False, title="Push to hub")
    peft: bool = Field(False, title="Use PEFT")
    quantization: Optional[str] = Field("int8", title="int4, int8, or None")
    lora_r: int = Field(16, title="LoRA-R")
    lora_alpha: int = Field(32, title="LoRA-Alpha")
    lora_dropout: float = Field(0.05, title="LoRA-Dropout")
    target_modules: str = Field("all-linear", title="Target modules for PEFT")
    log: str = Field("none", title="Logging using experiment tracking")
    early_stopping_patience: int = Field(5, title="Early stopping patience")
    early_stopping_threshold: float = Field(0.01, title="Early stopping threshold")
