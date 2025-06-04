from typing import Optional

from pydantic import Field

from autotrain.trainers.common import AutoTrainParams


class TextClassificationParams(AutoTrainParams):
    """
    [`TextClassificationParams`] is a configuration class for text classification training parameters.

    Attributes:
        data_path (str): Path to the dataset.
        model (str): Name of the model to use. Default is "bert-base-uncased".
        lr (float): Learning rate. Default is 5e-5.
        epochs (int): Number of training epochs. Default is 3.
        max_seq_length (int): Maximum sequence length. Default is 128.
        batch_size (int): Training batch size. Default is 8.
        warmup_ratio (float): Warmup proportion. Default is 0.1.
        gradient_accumulation (int): Number of gradient accumulation steps. Default is 1.
        optimizer (str): Optimizer to use. Default is "adamw_torch".
        scheduler (str): Scheduler to use. Default is "linear".
        weight_decay (float): Weight decay. Default is 0.0.
        max_grad_norm (float): Maximum gradient norm. Default is 1.0.
        seed (int): Random seed. Default is 42.
        train_split (str): Name of the training split. Default is "train".
        valid_split (Optional[str]): Name of the validation split. Default is None.
        text_column (str): Name of the text column in the dataset. Default is "text".
        target_column (str): Name of the target column in the dataset. Default is "target".
        logging_steps (int): Number of steps between logging. Default is -1.
        project_name (str): Name of the project. Default is "project-name".
        auto_find_batch_size (bool): Whether to automatically find the batch size. Default is False.
        mixed_precision (Optional[str]): Mixed precision setting (fp16, bf16, or None). Default is None.
        save_total_limit (int): Total number of checkpoints to save. Default is 1.
        token (Optional[str]): Hub token for authentication. Default is None.
        push_to_hub (bool): Whether to push the model to the hub. Default is False.
        eval_strategy (str): Evaluation strategy. Default is "epoch".
        username (Optional[str]): Hugging Face username. Default is None.
        log (str): Logging method for experiment tracking. Default is "none".
        early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped. Default is 5.
        early_stopping_threshold (float): Threshold for measuring the new optimum to continue training. Default is 0.01.
    """

    data_path: str = Field(None, title="Data path")
    model: str = Field("bert-base-uncased", title="Model name")
    lr: float = Field(5e-5, title="Learning rate")
    epochs: int = Field(3, title="Number of training epochs")
    max_seq_length: int = Field(128, title="Max sequence length")
    batch_size: int = Field(8, title="Training batch size")
    warmup_ratio: float = Field(0.1, title="Warmup proportion")
    gradient_accumulation: int = Field(1, title="Gradient accumulation steps")
    optimizer: str = Field("adamw_torch", title="Optimizer")
    scheduler: str = Field("linear", title="Scheduler")
    weight_decay: float = Field(0.0, title="Weight decay")
    max_grad_norm: float = Field(1.0, title="Max gradient norm")
    seed: int = Field(42, title="Seed")
    train_split: str = Field("train", title="Train split")
    valid_split: Optional[str] = Field(None, title="Validation split")
    text_column: str = Field("text", title="Text column")
    target_column: str = Field("target", title="Target column")
    logging_steps: int = Field(-1, title="Logging steps")
    project_name: str = Field("project-name", title="Output directory")
    auto_find_batch_size: bool = Field(False, title="Auto find batch size")
    mixed_precision: Optional[str] = Field(None, title="fp16, bf16, or None")
    save_total_limit: int = Field(1, title="Save total limit")
    token: Optional[str] = Field(None, title="Hub Token")
    push_to_hub: bool = Field(False, title="Push to hub")
    eval_strategy: str = Field("epoch", title="Evaluation strategy")
    username: Optional[str] = Field(None, title="Hugging Face Username")
    log: str = Field("none", title="Logging using experiment tracking")
    early_stopping_patience: int = Field(5, title="Early stopping patience")
    early_stopping_threshold: float = Field(0.01, title="Early stopping threshold")
