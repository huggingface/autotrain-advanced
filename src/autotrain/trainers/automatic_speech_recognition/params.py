from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from autotrain.app.models import fetch_models
import os
import logging
from autotrain.trainers.common import AutoTrainParams

logger = logging.getLogger(__name__)

class AutomaticSpeechRecognitionParams(AutoTrainParams):
    """
    Parameters for automatic speech recognition training.

    Attributes:
        project_name (str): Name of the project.
        data_path (str): Path to the data directory or hub dataset name.
        model (str): Name of the model to use.
        username (str): HuggingFace username.
        token (Optional[str]): HuggingFace token for authentication.
        using_hub_dataset (bool): Whether to use a dataset from the HuggingFace Hub.
        train_split (Optional[str]): Name of the training split in the hub dataset.
        valid_split (Optional[str]): Name of the validation split in the hub dataset.
        audio_column (str): Name of the column containing audio data.
        text_column (str): Name of the column containing text data.
        max_duration (float): Maximum duration of audio in seconds.
        sampling_rate (int): Target sampling rate for audio.
        max_seq_length (int): Maximum sequence length for text.
        output_dir (str): Directory to save the model.
        batch_size (int): Batch size for training.
        gradient_accumulation (int): Number of steps to accumulate gradients.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        scheduler (str): Learning rate scheduler type.
        optimizer (str): Optimizer type.
        mixed_precision (str): Mixed precision training type.
        push_to_hub (bool): Whether to push the model to the HuggingFace Hub.
        hub_model_id (Optional[str]): ID of the model on the HuggingFace Hub.
        log (str): Logging method for experiment tracking.
        logging_steps (int): Number of steps between logging.
        save_total_limit (int): Maximum number of checkpoints to keep.
        warmup_ratio (float): Warmup ratio for learning rate scheduler.
        weight_decay (float): Weight decay for the optimizer.
        max_grad_norm (float): Maximum gradient norm for clipping.
        seed (int): Random seed for reproducibility.
        early_stopping_patience (int): Number of epochs with no improvement for early stopping.
        early_stopping_threshold (float): Threshold for early stopping improvement.
        auto_find_batch_size (bool): Automatically find optimal batch size.
        eval_strategy (str): Evaluation strategy during training.
    """
    
    project_name: str
    data_path: str
    model: str
    username: str
    token: Optional[str] = None
    
    
    using_hub_dataset: bool = False
    train_split: Optional[str] = None
    valid_split: Optional[str] = None
    audio_column: str = "audio"
    text_column: str = "transcription"
    max_duration: float = 30.0
    sampling_rate: int = 16000
    max_seq_length: int = 128
    
    
    output_dir: str = Field(default="output")
    
    
    batch_size: int = 8
    gradient_accumulation: int = 1
    epochs: int = 3
    lr: float = 3e-5
    scheduler: str = "linear"
    optimizer: str = "adamw_torch"
    mixed_precision: str = "no"
    
    
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    
    
    log: str = Field("tensorboard", title="Logging method for experiment tracking")
    
    
    logging_steps: int = Field(10, title="Number of steps between logging")
    save_total_limit: int = Field(2, title="Maximum number of checkpoints to keep")
    warmup_ratio: float = Field(0.1, title="Warmup ratio for learning rate scheduler")
    weight_decay: float = Field(0.01, title="Weight decay for the optimizer")
    max_grad_norm: float = Field(1.0, title="Maximum gradient norm for clipping")
    seed: int = Field(42, title="Random seed for reproducibility")
    early_stopping_patience: int = Field(3, title="Number of epochs with no improvement for early stopping")
    early_stopping_threshold: float = Field(0.01, title="Threshold for early stopping improvement")
    auto_find_batch_size: bool = Field(False, title="Automatically find optimal batch size")
    eval_strategy: str = Field("epoch", title="Evaluation strategy during training")
    
    def validate_params(self):
        """Validate parameters."""
       
        available_models = fetch_models()
        model_found = False
        for category, models in available_models.items():
            if self.model in models:
                model_found = True
                break
        if not model_found:
            raise ValueError(f"Model {self.model} not found in available models: {available_models}")
        
        
        if not self.data_path:
            raise ValueError("data_path must be provided")
            
        
        if self.using_hub_dataset:
            if not self.train_split:
                raise ValueError("train_split must be provided when using hub dataset")
            if not self.token:
                raise ValueError("token must be provided when using hub dataset")
        else:
            
            if not os.path.exists(self.data_path):
                raise ValueError(f"data_path does not exist: {self.data_path}")
            
            
            files = os.listdir(self.data_path)
            
           
            csv_files = [f for f in files if f.endswith('.csv')]
            if not csv_files:
                raise ValueError(f"No CSV file found in {self.data_path}")
            
           
            audio_folder = os.path.join(self.data_path, 'audio')
            if not os.path.exists(audio_folder):
                logger.warning(f"Audio folder not found: {audio_folder}")
                logger.warning("Will try to use audio paths as provided in CSV")
                
       
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation <= 0:
            raise ValueError("gradient_accumulation must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.lr <= 0:
            raise ValueError("learning rate must be positive")
            
        
        if self.mixed_precision not in ["no", "fp16", "bf16"]:
            raise ValueError("mixed_precision must be one of: no, fp16, bf16")
            
       
        if self.scheduler not in ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]:
            raise ValueError("scheduler must be one of: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup")
            
        
        if self.optimizer not in ["adamw_hf", "adamw_torch", "adamw_torch_fused", "adamw_apex_fused", "adafactor", "adamw_anyprecision", "sgd", "adagrad"]:
            raise ValueError("optimizer must be one of: adamw_hf, adamw_torch, adamw_torch_fused, adamw_apex_fused, adafactor, adamw_anyprecision, sgd, adagrad")