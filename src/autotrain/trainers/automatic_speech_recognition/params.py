from typing import Optional
from pydantic import BaseModel, Field
from autotrain.app.models import fetch_models
from autotrain.trainers.common import AutoTrainParams

class AutomaticSpeechRecognitionParams(AutoTrainParams):
    """
    AutomaticSpeechRecognitionParams is a configuration class for ASR (Automatic Speech Recognition) training parameters.
    
    Attributes:
        data_path (str): Path to the dataset.
        model (str): Pre-trained model name or path (e.g., "facebook/wav2vec2-base-960h").
        username (Optional[str]): Hugging Face account username.
        lr (float): Learning rate for the optimizer. Default is 3e-4.
        epochs (int): Number of epochs for training. Default is 3.
        batch_size (int): Batch size for training. Default is 8.
        warmup_ratio (float): Warmup ratio for learning rate scheduler. Default is 0.1.
        gradient_accumulation (int): Number of gradient accumulation steps. Default is 1.
        optimizer (str): Optimizer type. Default is "adamw_torch".
        scheduler (str): Learning rate scheduler type. Default is "linear".
        weight_decay (float): Weight decay for the optimizer. Default is 0.01.
        max_grad_norm (float): Maximum gradient norm for clipping. Default is 1.0.
        seed (int): Random seed for reproducibility. Default is 42.
        train_split (str): Name of the training data split. Default is "train".
        valid_split (Optional[str]): Name of the validation data split.
        logging_steps (int): Number of steps between logging. Default is -1.
        project_name (str): Name of the project for output directory. Default is "project-name".
        auto_find_batch_size (bool): Automatically find optimal batch size. Default is False.
        mixed_precision (Optional[str]): Mixed precision training mode (fp16, bf16, or None).
        save_total_limit (int): Maximum number of checkpoints to keep. Default is 1.
        token (Optional[str]): Hugging Face Hub token for authentication.
        push_to_hub (bool): Whether to push the model to Hugging Face Hub. Default is False.
        eval_strategy (str): Evaluation strategy during training. Default is "epoch".
        audio_column (str): Column name for audio data in the dataset. Default is "audio".
        text_column (str): Column name for transcription/labels in the dataset. Default is "transcription".
        max_duration (float): Maximum audio duration (in seconds) for training samples. Default is 30.0.
        sampling_rate (int): Audio sampling rate. Default is 16000.
        max_seq_length (int): Maximum sequence length for text labels. Default is 128.
        log (str): Logging method for experiment tracking. Default is "none".
        early_stopping_patience (int): Number of epochs with no improvement for early stopping. Default is 5.
        early_stopping_threshold (float): Threshold for early stopping. Default is 0.01.
    """

    data_path: str = Field(None, title="Path to the dataset")
    model: str = Field("facebook/wav2vec2-base-960h", title="Pre-trained model name or path")
    username: Optional[str] = Field(None, title="Hugging Face account username")
    lr: float = Field(3e-4, title="Learning rate")
    epochs: int = Field(3, title="Number of training epochs")
    batch_size: int = Field(8, title="Batch size")
    warmup_ratio: float = Field(0.1, title="Warmup ratio for scheduler")
    gradient_accumulation: int = Field(1, title="Gradient accumulation steps")
    optimizer: str = Field("adamw_torch", title="Optimizer type")
    scheduler: str = Field("linear", title="Scheduler type")
    weight_decay: float = Field(0.01, title="Weight decay")
    max_grad_norm: float = Field(1.0, title="Max gradient norm for clipping")
    seed: int = Field(42, title="Random seed")
    train_split: str = Field("train", title="Training split name")
    valid_split: Optional[str] = Field(None, title="Validation split name")
    logging_steps: int = Field(-1, title="Steps between logging")
    project_name: str = Field("project-name", title="Project/output directory name")
    auto_find_batch_size: bool = Field(False, title="Auto batch size search")
    mixed_precision: Optional[str] = Field(None, title="Mixed precision mode (fp16, bf16, or None)")
    save_total_limit: int = Field(1, title="Max checkpoints to keep")
    token: Optional[str] = Field(None, title="Hugging Face Hub token")
    push_to_hub: bool = Field(False, title="Push model to Hugging Face Hub")
    eval_strategy: str = Field("epoch", title="Evaluation strategy")
    audio_column: str = Field("audio", title="Audio column name")
    text_column: str = Field("transcription", title="Transcription/label column name")
    max_duration: float = Field(30.0, title="Max audio duration (seconds)")
    sampling_rate: int = Field(16000, title="Audio sampling rate")
    max_seq_length: int = Field(448, title="Max sequence length for labels")
    log: str = Field("none", title="Experiment logging method")
    early_stopping_patience: int = Field(5, title="Early stopping patience (epochs)")
    early_stopping_threshold: float = Field(0.01, title="Early stopping threshold")