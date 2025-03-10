from dataclasses import dataclass, field
from typing import List, Optional

from peft import LoraConfig

@dataclass
class WhisperTrainingParams:
    """Parameters for Whisper ASR training.
    
    This class defines all parameters needed for fine-tuning Whisper models for Automatic Speech Recognition (ASR),
    including audio processing settings, model configuration, training hyperparameters, and PEFT/LoRA settings.
    
    Attributes:
        sampling_rate (int): Audio sampling rate in Hz. Default is 16000.
        max_duration_secs (float): Maximum duration of audio clips in seconds. Longer clips will be truncated. Default is 30.0.
        preprocessing_num_workers (int): Number of worker processes for audio preprocessing. Default is 4.
        
        model_name (str): Name or path of the Whisper model to fine-tune. Default is "openai/whisper-small".
        language (str): Language code for ASR (e.g., "en" for English). Default is "en".
        task (str): ASR task type ("transcribe" or "translate"). Default is "transcribe".
        
        learning_rate (float): Learning rate for training. Default is 5e-5.
        num_train_epochs (int): Number of training epochs. Default is 3.
        per_device_train_batch_size (int): Training batch size per device. Default is 8.
        per_device_eval_batch_size (int): Evaluation batch size per device. Default is 8.
        gradient_accumulation_steps (int): Number of steps for gradient accumulation. Default is 1.
        eval_steps (int): Number of steps between evaluations. Default is 100.
        save_steps (int): Number of steps between model checkpoints. Default is 500.
        logging_steps (int): Number of steps between logging updates. Default is 10.
        max_steps (Optional[int]): Maximum number of training steps. If None, train for num_train_epochs. Default is None.
        warmup_steps (int): Number of warmup steps for learning rate scheduler. Default is 0.
        mixed_precision (str): Mixed precision training type. Default is "fp16".
        log (str): Logging type. Default is "tensorboard".
        
        use_peft (bool): Whether to use PEFT/LoRA for efficient fine-tuning. Default is True.
        lora_r (int): Rank of LoRA matrices. Default is 8.
        lora_alpha (int): LoRA alpha parameter for scaling updates. Default is 32.
        lora_dropout (float): Dropout probability for LoRA layers. Default is 0.1.
        target_modules (List[str]): List of model modules to apply LoRA to. Defaults to attention and feed-forward layers.
    """
    # Audio processing parameters
    sampling_rate: int = 16000
    max_duration_secs: float = 30.0
    preprocessing_num_workers: int = 4
    
    # Model parameters
    model_name: str = "openai/whisper-small"
    language: str = "en"
    task: str = "transcribe"
    
    # Training parameters
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10
    max_steps: Optional[int] = None
    warmup_steps: int = 0
    mixed_precision: str = "fp16"
    log: str = "tensorboard"
    
    # PEFT/LoRA parameters
    use_peft: bool = True
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be positive, got {self.sampling_rate}")
        
        if self.max_duration_secs <= 0:
            raise ValueError(f"max_duration_secs must be positive, got {self.max_duration_secs}")
        
        if self.preprocessing_num_workers < 0:
            raise ValueError(f"preprocessing_num_workers must be non-negative, got {self.preprocessing_num_workers}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.num_train_epochs <= 0:
            raise ValueError(f"num_train_epochs must be positive, got {self.num_train_epochs}")
        
        if self.per_device_train_batch_size <= 0:
            raise ValueError(f"per_device_train_batch_size must be positive, got {self.per_device_train_batch_size}")
        
        if self.per_device_eval_batch_size <= 0:
            raise ValueError(f"per_device_eval_batch_size must be positive, got {self.per_device_eval_batch_size}")
        
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}")
        
        if self.eval_steps <= 0:
            raise ValueError(f"eval_steps must be positive, got {self.eval_steps}")
        
        if self.save_steps <= 0:
            raise ValueError(f"save_steps must be positive, got {self.save_steps}")
        
        if self.logging_steps <= 0:
            raise ValueError(f"logging_steps must be positive, got {self.logging_steps}")
        
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive if set, got {self.max_steps}")
        
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        
        if self.mixed_precision not in ["no", "fp16", "bf16"]:
            raise ValueError(f"mixed_precision must be one of ['no', 'fp16', 'bf16'], got {self.mixed_precision}")
        
        if self.use_peft:
            if self.lora_r <= 0:
                raise ValueError(f"lora_r must be positive when use_peft=True, got {self.lora_r}")
            
            if self.lora_alpha <= 0:
                raise ValueError(f"lora_alpha must be positive when use_peft=True, got {self.lora_alpha}")
            
            if not 0 <= self.lora_dropout <= 1:
                raise ValueError(f"lora_dropout must be between 0 and 1 when use_peft=True, got {self.lora_dropout}")
            
            if not self.target_modules:
                raise ValueError("target_modules cannot be empty when use_peft=True")
    
    def get_lora_config(self) -> Optional[LoraConfig]:
        """Returns LoRA configuration if PEFT is enabled.
        
        Returns:
            Optional[LoraConfig]: LoRA configuration for PEFT if use_peft is True, None otherwise.
        """
        if not self.use_peft:
            return None
            
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        ) 