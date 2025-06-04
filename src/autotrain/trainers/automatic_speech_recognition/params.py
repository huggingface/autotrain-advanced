from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from autotrain.app.models import fetch_models
import os
import logging

logger = logging.getLogger(__name__)

class AutomaticSpeechRecognitionParams(BaseModel):
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
    """
    # Base fields
    project_name: str
    data_path: str
    model: str
    username: str
    token: Optional[str] = None
    
    # Dataset fields
    using_hub_dataset: bool = False
    train_split: Optional[str] = None
    valid_split: Optional[str] = None
    audio_column: str = "audio"
    text_column: str = "transcription"
    max_duration: float = 30.0
    sampling_rate: int = 16000
    max_seq_length: int = 128
    
    # Output fields
    output_dir: str = Field(default="output")
    
    # Training fields
    batch_size: int = 8
    gradient_accumulation: int = 1
    epochs: int = 3
    lr: float = 3e-5
    scheduler: str = "linear"
    optimizer: str = "adamw_torch"
    mixed_precision: str = "no"
    
    # Hub fields
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    
    def validate_params(self):
        """Validate parameters."""
        # Validate model
        available_models = fetch_models()
        model_found = False
        for category, models in available_models.items():
            if self.model in models:
                model_found = True
                break
        if not model_found:
            raise ValueError(f"Model {self.model} not found in available models: {available_models}")
        
        # Validate data path
        if not self.data_path:
            raise ValueError("data_path must be provided")
            
        # Validate hub dataset parameters
        if self.using_hub_dataset:
            if not self.train_split:
                raise ValueError("train_split must be provided when using hub dataset")
            if not self.token:
                raise ValueError("token must be provided when using hub dataset")
        else:
            # Validate local dataset parameters
            if not os.path.exists(self.data_path):
                raise ValueError(f"data_path does not exist: {self.data_path}")
            
            # Check for data files
            files = os.listdir(self.data_path)
            
            # Check for CSV file
            csv_files = [f for f in files if f.endswith('.csv')]
            if not csv_files:
                raise ValueError(f"No CSV file found in {self.data_path}")
            
            # Check for audio folder
            audio_folder = os.path.join(self.data_path, 'audio')
            if not os.path.exists(audio_folder):
                logger.warning(f"Audio folder not found: {audio_folder}")
                logger.warning("Will try to use audio paths as provided in CSV")
                
        # Validate training parameters
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation <= 0:
            raise ValueError("gradient_accumulation must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.lr <= 0:
            raise ValueError("learning rate must be positive")
            
        # Validate mixed precision
        if self.mixed_precision not in ["no", "fp16", "bf16"]:
            raise ValueError("mixed_precision must be one of: no, fp16, bf16")
            
        # Validate scheduler
        if self.scheduler not in ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]:
            raise ValueError("scheduler must be one of: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup")
            
        # Validate optimizer
        if self.optimizer not in ["adamw_hf", "adamw_torch", "adamw_torch_fused", "adamw_apex_fused", "adafactor", "adamw_anyprecision", "sgd", "adagrad"]:
            raise ValueError("optimizer must be one of: adamw_hf, adamw_torch, adamw_torch_fused, adamw_apex_fused, adafactor, adamw_anyprecision, sgd, adagrad") 