from typing import Optional

from pydantic import Field

from autotrain.trainers.common import AutoTrainParams


class SentenceTransformersParams(AutoTrainParams):
    """
    SentenceTransformersParams is a configuration class for setting up parameters for training sentence transformers.

    Attributes:
        data_path (str): Path to the dataset.
        model (str): Name of the pre-trained model to use. Default is "microsoft/mpnet-base".
        lr (float): Learning rate for training. Default is 3e-5.
        epochs (int): Number of training epochs. Default is 3.
        max_seq_length (int): Maximum sequence length for the input. Default is 128.
        batch_size (int): Batch size for training. Default is 8.
        warmup_ratio (float): Proportion of training to perform learning rate warmup. Default is 0.1.
        gradient_accumulation (int): Number of steps to accumulate gradients before updating. Default is 1.
        optimizer (str): Optimizer to use. Default is "adamw_torch".
        scheduler (str): Learning rate scheduler to use. Default is "linear".
        weight_decay (float): Weight decay to apply. Default is 0.0.
        max_grad_norm (float): Maximum gradient norm for clipping. Default is 1.0.
        seed (int): Random seed for reproducibility. Default is 42.
        train_split (str): Name of the training data split. Default is "train".
        valid_split (Optional[str]): Name of the validation data split. Default is None.
        logging_steps (int): Number of steps between logging. Default is -1.
        project_name (str): Name of the project for output directory. Default is "project-name".
        auto_find_batch_size (bool): Whether to automatically find the optimal batch size. Default is False.
        mixed_precision (Optional[str]): Mixed precision training mode (fp16, bf16, or None). Default is None.
        save_total_limit (int): Maximum number of checkpoints to save. Default is 1.
        token (Optional[str]): Token for accessing Hugging Face Hub. Default is None.
        push_to_hub (bool): Whether to push the model to Hugging Face Hub. Default is False.
        eval_strategy (str): Evaluation strategy to use. Default is "epoch".
        username (Optional[str]): Hugging Face username. Default is None.
        log (str): Logging method for experiment tracking. Default is "none".
        early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped. Default is 5.
        early_stopping_threshold (float): Threshold for measuring the new optimum, to qualify as an improvement. Default is 0.01.
        trainer (str): Name of the trainer to use. Default is "pair_score".
        sentence1_column (str): Name of the column containing the first sentence. Default is "sentence1".
        sentence2_column (str): Name of the column containing the second sentence. Default is "sentence2".
        sentence3_column (Optional[str]): Name of the column containing the third sentence (if applicable). Default is None.
        target_column (Optional[str]): Name of the column containing the target variable. Default is None.
    """

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
    eval_strategy: str = Field("epoch", title="Evaluation strategy")
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
    sentence3_column: Optional[str] = Field(None, title="Sentence 3 column")
    target_column: Optional[str] = Field(None, title="Target column")
