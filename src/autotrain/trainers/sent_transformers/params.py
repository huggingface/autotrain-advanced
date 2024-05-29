from typing import Optional

from pydantic import Field

from autotrain.trainers.common import AutoTrainParams


class SentenceTransformersParams(AutoTrainParams):
    data_path: str = Field(None, title="Data path")
    model: str = Field("microsoft/mpnet-base", title="Model name")
    lr: float = Field(3e-5, title="Learning rate")
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
    logging_steps: int = Field(-1, title="Logging steps")
    project_name: str = Field("project-name", title="Output directory")
    auto_find_batch_size: bool = Field(False, title="Auto find batch size")
    mixed_precision: Optional[str] = Field(None, title="fp16, bf16, or None")
    save_total_limit: int = Field(1, title="Save total limit")
    token: Optional[str] = Field(None, title="Hub Token")
    push_to_hub: bool = Field(False, title="Push to hub")
    evaluation_strategy: str = Field("epoch", title="Evaluation strategy")
    username: Optional[str] = Field(None, title="Hugging Face Username")
    log: str = Field("none", title="Logging using experiment tracking")
    early_stopping_patience: int = Field(5, title="Early stopping patience")
    early_stopping_threshold: float = Field(0.01, title="Early stopping threshold")
    # trainers: pair, pair_class, pair_score, triplet, qa
    # pair: sentence1, sentence2
    # pair_class: sentence1, sentence2, target
    # pair_score: sentence1, sentence2, target
    # triplet: sentence1, sentence2, sentence3
    # qa: sentence1, sentence2
    trainer: str = Field("pair_score", title="Trainer name")
    sentence1_column: str = Field("sentence1", title="Sentence 1 column")
    sentence2_column: str = Field("sentence2", title="Sentence 2 column")
    sentence3_column: Optional[str] = Field("sentence3", title="Sentence 3 column")
    target_column: Optional[str] = Field("target", title="Target column")
