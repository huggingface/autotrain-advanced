import argparse
import json
import os
from typing import Optional, Dict, Any
import torch
import librosa
from datetime import datetime
import pandas as pd
import traceback
import sys
import glob
import shutil
import logging
import functools
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autotrain.asr.debug")
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("accelerate").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

from accelerate.state import PartialState
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoModelForCTC,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoModel,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback


from autotrain import logger
from autotrain.trainers.common import (
    ALLOW_REMOTE_CODE,
    LossLoggingCallback,
    TrainStartCallback,
    UploadLogs,
    monitor,
    pause_space,
    remove_autotrain_data,
    save_training_params,
)
from autotrain.trainers.automatic_speech_recognition.dataset import AutoTrainASRDataset, load_life_app_dataset
from autotrain.trainers.automatic_speech_recognition import utils
from autotrain.trainers.automatic_speech_recognition.params import AutomaticSpeechRecognitionParams
from autotrain.trainers.automatic_speech_recognition.utils import compute_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()

def dynamic_padding_collator(batch):
    if "input_features" in batch[0]:
        # Whisper: pad/truncate to [80, 3000]
        target_length = 3000
        padded_features = []
        for feat in [item["input_features"] for item in batch]:
            if feat.shape[1] < target_length:
                padding = torch.zeros(feat.shape[0], target_length - feat.shape[1], dtype=feat.dtype)
                padded_feat = torch.cat([feat, padding], dim=1)
            else:
                padded_feat = feat[:, :target_length]
            padded_features.append(padded_feat)
        input_features = torch.stack(padded_features)
        labels = [item["labels"] for item in batch]
        return {
            "input_features": input_features,
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
        }
    elif "input_values" in batch[0]:
        input_values = [item["input_values"] for item in batch]
        labels = [item["labels"] for item in batch]
        return {
            "input_values": torch.stack(input_values),
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
        }
    else:
        raise ValueError("Batch items must have either 'input_features' or 'input_values'.")

def load_data(params, is_validation=False):
    """
    Load dataset from local directory or HuggingFace Hub.

    Args:
        params: Training parameters
        is_validation: Whether to load validation data
        
    Returns:
        Dataset: Loaded dataset
    """
    try:
        split = params.valid_split if is_validation else params.train_split
        if params.data_path == f"{params.project_name}/autotrain-data":
            dataset = load_from_disk(params.data_path)[split]
            logger.info(f"Loaded {'validation' if is_validation else 'train'} dataset from disk with {len(dataset)} examples.")
            return dataset
        if ":" in split:
            dataset_config_name, split_name = split.split(":")
            dataset = load_dataset(
                params.data_path,
                name=dataset_config_name,
                split=split_name,
                token=params.token if params.token else None,
                trust_remote_code=ALLOW_REMOTE_CODE,
            )
            logger.info(f"Loaded {'validation' if is_validation else 'train'} dataset from hub with {len(dataset)} examples.")
            return dataset
        if split:
            dataset = load_dataset(
                params.data_path,
                split=split,
                token=params.token if params.token else None,
                trust_remote_code=ALLOW_REMOTE_CODE,
            )
            logger.info(f"Loaded {'validation' if is_validation else 'train'} dataset from hub with {len(dataset)} examples.")
            return dataset
        # If none of the above, use the robust local CSV/audio loader (previously dead code)
        if not os.path.exists(params.data_path):
            logger.error(f"Data path does not exist: {params.data_path}")
            raise ValueError(f"Data path does not exist: {params.data_path}")
        files = os.listdir(params.data_path)
        logger.info(f"Files in directory: {files}")
        if 'train' in files and 'validation' in files:
            logger.info("Detected HuggingFace dataset format")
            if is_validation:
                dataset = load_from_disk(os.path.join(params.data_path, 'validation'))
            else:
                dataset = load_from_disk(os.path.join(params.data_path, 'train'))
            logger.info(f"Loaded {'validation' if is_validation else 'train'} dataset with {len(dataset)} examples")
            return dataset
        csv_files = [f for f in files if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(f"No CSV file found in {params.data_path}")
        csv_file_path = os.path.join(params.data_path, csv_files[0])
        logger.info(f"Using CSV file: {csv_file_path}")
        logger.info(f"Loading CSV file: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        logger.info(f"CSV columns: {df.columns.tolist()}")
        logger.info(f"Number of examples: {len(df)}")
        if 'audio' in df.columns:
            logger.info("Mapping audio to audio")
            df = df.rename(columns={'audio': 'audio'})
        if 'transcription' in df.columns:
            logger.info("Mapping transcription to transcription")
            df = df.rename(columns={'transcription': 'transcription'})
        required_columns = ['audio', 'transcription']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}. Available columns: {df.columns.tolist()}")
        if is_validation:
            valid_size = int(len(df) * 0.2)
            df = df.sample(n=valid_size, random_state=42)
            logger.info(f"Using {valid_size} examples for validation")
        else:
            train_size = int(len(df) * 0.8)
            df = df.sample(n=train_size, random_state=42)
            logger.info(f"Using {train_size} examples for training")
        invalid_audio_files = []
        audio_folder = os.path.join(params.data_path, 'audio')
        if not os.path.exists(audio_folder):
            logger.warning(f"Audio folder not found: {audio_folder}")
            logger.warning("Will try to use audio paths as provided in CSV")
        for idx, audio_path in enumerate(df['audio']):
            if not os.path.isabs(audio_path):
                if os.path.exists(audio_folder):
                    audio_file = os.path.join(audio_folder, os.path.basename(audio_path))
                    if os.path.exists(audio_file):
                        df.at[idx, 'audio'] = os.path.abspath(audio_file)
                        continue
                audio_file = os.path.join(params.data_path, audio_path)
                if os.path.exists(audio_file):
                    df.at[idx, 'audio'] = os.path.abspath(audio_file)
                    continue
            if not os.path.exists(audio_path):
                invalid_audio_files.append((idx, audio_path))
        if invalid_audio_files:
            error_msg = "The following audio files were not found:\n"
            for idx, path in invalid_audio_files[:5]:
                error_msg += f"Row {idx}: {path}\n"
            if len(invalid_audio_files) > 5:
                error_msg += f"... and {len(invalid_audio_files) - 5} more files"
            raise ValueError(error_msg)
        dataset = Dataset.from_pandas(df)
        if len(dataset) > 0:
            logger.info("First example in dataset:")
            logger.info(f"Audio path: {dataset[0]['audio']}")
            logger.info(f"Transcription: {dataset[0]['transcription']}")
            if 'duration' in dataset[0]:
                logger.info(f"Duration: {dataset[0]['duration']}")
            try:
                audio, sr = librosa.load(dataset[0]['audio'], sr=params.sampling_rate)
                logger.info(f"Successfully loaded first audio file. Duration: {len(audio)/sr:.2f}s")
            except Exception as e:
                logger.error(f"Error loading first audio file: {str(e)}")
                raise
        return dataset
            
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def load_model_and_processor(params):
    """
    Load model and processor for ASR training.
    
    Args:
        params: Training parameters
        
    Returns:
        tuple: (model, processor)
    """
    try:
        
        logger.info(f"Starting processor load for: {params.model}")
        try:
            processor = AutoProcessor.from_pretrained(
                params.model,
                token=params.token if params.token else None,
                trust_remote_code=ALLOW_REMOTE_CODE,
            )
        except Exception as e:
            logger.warning(f"Could not load processor with AutoProcessor, trying Wav2Vec2Processor...")
            try:
                from transformers import Wav2Vec2Processor
                processor = Wav2Vec2Processor.from_pretrained(
                    params.model,
                    token=params.token if params.token else None,
                    trust_remote_code=ALLOW_REMOTE_CODE,
                )
            except Exception as e2:
                logger.warning(f"Could not load processor with Wav2Vec2Processor, trying WhisperProcessor...")
                try:
                    from transformers import WhisperProcessor
                    processor = WhisperProcessor.from_pretrained(
                        params.model,
                        token=params.token if params.token else None,
                        trust_remote_code=ALLOW_REMOTE_CODE,
                    )
                except Exception as e3:
                    raise ValueError(f"Could not load processor with any known processor type: {str(e3)}")
        
        
        model_name = params.model.lower()
        
        
        try:
            logger.info("Attempting to load as Seq2Seq model...")
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                params.model,
                trust_remote_code=ALLOW_REMOTE_CODE,
            )
            logger.info("Successfully loaded as Seq2Seq model")
        except Exception as e:
            logger.info(f"Could not load as Seq2Seq model: {str(e)}")
            
            try:
                logger.info("Attempting to load as CTC model...")
                model = AutoModelForCTC.from_pretrained(
                    params.model,
                    trust_remote_code=ALLOW_REMOTE_CODE,
                )
                logger.info("Successfully loaded as CTC model")
            except Exception as e:
                logger.info(f"Could not load as CTC model: {str(e)}")
                
                try:
                    logger.info("Attempting to load as generic model...")
                    model = AutoModel.from_pretrained(
                        params.model,
                        trust_remote_code=ALLOW_REMOTE_CODE,
                    )
                    logger.info("Successfully loaded as generic model")
                except Exception as e:
                    raise ValueError(f"Could not load model {params.model} as any supported type: {str(e)}")
        
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Moving model to device: {device}")
        model = model.to(device)
        logger.info(f"Model moved to device: {device}")
        
        
        logger.info(f"Model type: {model.__class__.__name__}")
        logger.info(f"Model configuration: {model.config}")
        
       
        required_methods = ['forward', 'get_encoder', 'get_decoder']
        missing_methods = [method for method in required_methods if not hasattr(model, method)]
        if missing_methods:
            logger.warning(f"Model is missing some required methods: {missing_methods}")
            logger.warning("This might affect training. Please check model documentation.")
        
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model, processor
        
    except Exception as e:
        logger.error(f"Error loading model and processor: {str(e)}")
        raise

@monitor
def train(config: Dict[str, Any]):
    try:
        if isinstance(config, dict):
            config = AutomaticSpeechRecognitionParams(**config)
        # Debug: Log all key training arguments
        logger.info(f"[DEBUG] Model name: {getattr(config, 'model', None)}")
        logger.info(f"[DEBUG] Training arguments: lr={getattr(config, 'lr', None)}, epochs={getattr(config, 'epochs', None)}, batch_size={getattr(config, 'batch_size', None)}, warmup_ratio={getattr(config, 'warmup_ratio', None)}, weight_decay={getattr(config, 'weight_decay', None)}, max_seq_length={getattr(config, 'max_seq_length', None)}, sampling_rate={getattr(config, 'sampling_rate', None)}")
        # Enforce push_to_hub defaults
        if not hasattr(config, "push_to_hub") or config.push_to_hub is None:
            config.push_to_hub = True
        logger.info("Initializing ASR training pipeline...")
        logger.info("Parameters parsed.")
        
        # Enhanced device detection and logging
        import torch
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()
        
        if cuda_available:
            device = "cuda"
            num_gpus = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if num_gpus > 0 else "Unknown"
            logger.info(f"CUDA available: {cuda_available}")
            logger.info(f"Number of GPUs: {num_gpus}")
            logger.info(f"GPU name: {gpu_name}")
        elif mps_available:
            device = "mps"
            logger.info(f"MPS (Apple Silicon) available: {mps_available}")
        else:
            device = "cpu"
            logger.info("No GPU detected, using CPU")
            
        logger.info(f"Using device: {device}")
        
        # Warning if using CPU when GPU might be expected
        if device == "cpu" and cuda_available:
            logger.warning("WARNING: CUDA is available but training will use CPU!")
            logger.warning("This might be due to accelerate configuration or environment issues.")
        elif device == "cpu" and not cuda_available:
            logger.warning("WARNING: No GPU detected. Training will be very slow on CPU!")
            logger.warning("Consider using a machine with GPU for faster training.")

        logger.info(f"Using data_path: {config.data_path}")
        logger.info(f"Using audio_column: {getattr(config, 'audio_column', None)}")
        logger.info(f"Using text_column: {getattr(config, 'text_column', None)}")

        train_path = os.path.join(config.data_path, "train")
        if os.path.exists(train_path):
            logger.info("Loading training dataset from disk...")
            from datasets import load_from_disk
            dataset = load_from_disk(train_path)
            logger.info(f"Training dataset loaded with {len(dataset)} examples.")
        else:
            logger.info("Loading dataset using load_data()...")
            dataset = load_data(config)
            logger.info(f"Dataset loaded with {len(dataset)} examples.")
        validation_path = os.path.join(config.data_path, "validation")
        if os.path.exists(validation_path):
            logger.info("Loading validation dataset from disk...")
            from datasets import load_from_disk
            valid_dataset = load_from_disk(validation_path)
            logger.info(f"Validation dataset loaded with {len(valid_dataset)} examples.")
        else:
            valid_dataset = None
        logger.info("Loading model and processor...")
        model, processor = load_model_and_processor(config)
        logger.info("Model and processor loaded.")
        logger.info(f"[DEBUG] Model type: {type(model).__name__}")
        logger.info(f"[DEBUG] Processor type: {type(processor).__name__}")
        
        # Add dropout for small datasets to prevent overfitting
        dataset_size = len(dataset)
        if dataset_size < 1000:
            logger.info(f"Adding dropout to model for small dataset ({dataset_size} examples)")
            # Add dropout to model if it has config
            if hasattr(model, 'config'):
                if hasattr(model.config, 'dropout'):
                    model.config.dropout = 0.3
                    logger.info(f"Set model dropout to 0.3")
                if hasattr(model.config, 'attention_dropout'):
                    model.config.attention_dropout = 0.2
                    logger.info(f"Set attention dropout to 0.2")
                if hasattr(model.config, 'activation_dropout'):
                    model.config.activation_dropout = 0.2
                    logger.info(f"Set activation dropout to 0.2")
        
        logger.info("Creating training dataset object...")
        train_dataset = AutoTrainASRDataset(
            data=dataset,
            processor=processor,
            model=model,
            audio_column=config.audio_column,
            text_column=config.text_column,
            max_duration=config.max_duration,
            sampling_rate=config.sampling_rate,
            max_seq_length=getattr(config, 'max_seq_length', 448),
        )
        logger.info(f"Training dataset object created with {len(train_dataset)} examples.")
        logger.info(f"Training dataset type: {type(train_dataset).__name__}")
        if valid_dataset is not None:
            logger.info("Creating validation dataset object...")
            valid_dataset_obj = AutoTrainASRDataset(
                data=valid_dataset,
                processor=processor,
                model=model,
                audio_column=config.audio_column,
                text_column=config.text_column,
                max_duration=config.max_duration,
                sampling_rate=config.sampling_rate,
                max_seq_length=getattr(config, 'max_seq_length', 448),
            )
            logger.info(f"Validation dataset object created with {len(valid_dataset_obj)} examples.")
            logger.info(f"Validation dataset type: {type(valid_dataset_obj).__name__}")
        logger.info("Initializing Trainer...")
        # Anti-overfitting and anti-underfitting measures
        dataset_size = len(train_dataset)
        model_complexity = "unknown"
        
        # Estimate model complexity based on model name
        if "whisper-tiny" in config.model.lower():
            model_complexity = "tiny"
        elif "whisper-base" in config.model.lower():
            model_complexity = "base"
        elif "whisper-small" in config.model.lower():
            model_complexity = "small"
        elif "whisper-medium" in config.model.lower():
            model_complexity = "medium"
        elif "whisper-large" in config.model.lower():
            model_complexity = "large"
        elif "wav2vec2-base" in config.model.lower():
            model_complexity = "base"
        elif "wav2vec2-large" in config.model.lower():
            model_complexity = "large"
        
        logger.info(f"Dataset size: {dataset_size} examples")
        logger.info(f"Model complexity: {model_complexity}")
        
        # Determine if we need anti-overfitting or anti-underfitting measures
        if dataset_size < 1000:  # Small dataset - risk of overfitting
            logger.warning(f"Small dataset detected ({dataset_size} examples). Applying anti-overfitting measures:")
            logger.warning(f"- Reducing learning rate by 10x")
            logger.warning(f"- Increasing weight decay")
            logger.warning(f"- Adding dropout")
            
            adjusted_lr = config.lr / 10.0
            adjusted_weight_decay = 0.1
            early_stopping_patience = 1  # Stop early for small datasets
            adjusted_epochs = min(config.epochs, 2)  # Limit epochs for small datasets
            logger.info(f"Using early stopping patience: {early_stopping_patience} (small dataset)")
            logger.info(f"Limiting epochs to: {adjusted_epochs} (anti-overfitting)")
            
        elif dataset_size > 5000 and model_complexity in ["tiny", "base"]:  # Large dataset with small model - risk of underfitting
            logger.warning(f"Large dataset ({dataset_size} examples) with {model_complexity} model detected. Applying anti-underfitting measures:")
            logger.warning(f"- Increasing learning rate by 2x")
            logger.warning(f"- Reducing weight decay")
            logger.warning(f"- Increasing training epochs")
            
            adjusted_lr = config.lr * 2.0
            adjusted_weight_decay = 0.001
            early_stopping_patience = 5  # More patience for underfitting
            adjusted_epochs = config.epochs + 2  # More epochs for underfitting
            logger.info(f"Using early stopping patience: {early_stopping_patience} (anti-underfitting)")
            logger.info(f"Increasing epochs to: {adjusted_epochs} (anti-underfitting)")
            
        else:  # Balanced case
            logger.info(f"Balanced dataset-model combination. Using standard parameters.")
            adjusted_lr = config.lr
            adjusted_weight_decay = config.weight_decay
            early_stopping_patience = 3
            adjusted_epochs = config.epochs
            logger.info(f"Using standard early stopping patience: {early_stopping_patience}")
            logger.info(f"Using standard epochs: {adjusted_epochs}")
            
        # Calculate logging_steps the same way as other tasks
        user_set_logging_steps = hasattr(config, 'logging_steps') and config.logging_steps != -1
        if user_set_logging_steps:
            logging_steps = config.logging_steps
            logger.info(f"Logging steps (from config): {logging_steps}")
        else:
            logging_steps = None  # Do not set, use Trainer default
            logger.info("Logging steps: using Trainer default (500)")
        
        # Set evaluation_strategy from config.evaluation_strategy if available, else 'no'
        eval_strategy = getattr(config, 'evaluation_strategy', None)
        if eval_strategy is None:
            eval_strategy = "epoch" if valid_dataset is not None else "no"
        
        # Use dict format like image classification for better control
        training_args = dict(
            output_dir=config.project_name,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation,
            learning_rate=adjusted_lr,
            num_train_epochs=adjusted_epochs,
            save_strategy=eval_strategy,
            disable_tqdm=False,  # enables the progress bar like other tasks
            evaluation_strategy=eval_strategy,  # use config or fallback
            load_best_model_at_end=True if valid_dataset is not None else False,
            report_to=config.log,
            auto_find_batch_size=config.auto_find_batch_size,
            lr_scheduler_type=config.scheduler,
            optim=config.optimizer,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,
            push_to_hub=config.push_to_hub,
            save_total_limit=config.save_total_limit,
            fp16=config.mixed_precision == "fp16",
            bf16=config.mixed_precision == "bf16",
            ddp_find_unused_parameters=False,
        )
        if logging_steps is not None:
            training_args["logging_steps"] = logging_steps
            
        # Add mixed precision settings
        if config.mixed_precision == "fp16":
            training_args["fp16"] = True
        if config.mixed_precision == "bf16":
            training_args["bf16"] = True
            
        # Add best practices from main repo
        training_args["ddp_find_unused_parameters"] = False
        training_args["remove_unused_columns"] = False
        
        # Add gradient checkpointing for memory efficiency
        training_args["gradient_checkpointing"] = True
        logger.info("Trainer arguments set. Initializing Trainer...")
        logger.info("========================================")
        logger.info("TRAINING CONFIGURATION")
        logger.info("========================================")
        logger.info(f"   - Model: {config.model}")
        logger.info(f"   - Batch size: {config.batch_size}")
        logger.info(f"   - Original learning rate: {config.lr}")
        logger.info(f"   - Adjusted learning rate: {adjusted_lr}")
        logger.info(f"   - Original epochs: {config.epochs}")
        logger.info(f"   - Adjusted epochs: {adjusted_epochs}")
        logger.info(f"   - Mixed precision: {config.mixed_precision}")
        logger.info(f"   - Max sequence length: {getattr(config, 'max_seq_length', 448)}")
        logger.info(f"   - Weight decay: {adjusted_weight_decay}")
        logger.info(f"   - Early stopping patience: {early_stopping_patience}")
        logger.info(f"   - Audio column: {config.audio_column}")
        logger.info(f"   - Text column: {config.text_column}")
        logger.info(f"   - Max duration: {getattr(config, 'max_duration', 30.0)}s")
        logger.info(f"   - Sampling rate: {getattr(config, 'sampling_rate', 16000)}Hz")
        logger.info(f"   - Training examples: {len(train_dataset)}")
        logger.info(f"   - Validation examples: {len(valid_dataset_obj) if valid_dataset else 0}")
        logger.info(f"   - Logging steps: {logging_steps}")
        logger.info(f"   - Gradient accumulation: {config.gradient_accumulation}")
        logger.info(f"   - Warmup ratio: {getattr(config, 'warmup_ratio', 0.1)}")
        logger.info("========================================")
            
        # Setup callbacks like image classification
        if valid_dataset is not None:
            early_stop = EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=getattr(config, 'early_stopping_threshold', 0.01),
            )
            callbacks_to_use = [early_stop]
        else:
            callbacks_to_use = []
        
        # Add standard callbacks
        callbacks_to_use.extend([UploadLogs(config=config), LossLoggingCallback(), TrainStartCallback()])
        
        # Add underfitting detection callback
        if valid_dataset is not None:
            from autotrain.trainers.automatic_speech_recognition.utils import UnderfittingDetectionCallback
            underfitting_callback = UnderfittingDetectionCallback()
            callbacks_to_use.append(underfitting_callback)
        
        # Convert dict to TrainingArguments
        args = TrainingArguments(**training_args)
        
        # Add dataset size to args for better suggestions (after TrainingArguments creation)
        args.train_dataset_size = len(train_dataset)
        
        trainer_args = dict(
            args=args,
            model=model,
            callbacks=callbacks_to_use,
            data_collator=dynamic_padding_collator,
            compute_metrics=functools.partial(compute_metrics, processor=processor) if valid_dataset is not None else None,
        )
        
        trainer = Trainer(
            **trainer_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset_obj if valid_dataset is not None else None,
        )
        trainer.remove_callback(PrinterCallback)
        logger.info("Trainer initialized. Starting training...")
        logger.info("Starting training loop...")
        logger.info("========================================")
        logger.info("TRAINING STARTED - Watch for progress logs")
        logger.info("========================================")
        
        trainer.train()
        
        logger.info("========================================")
        logger.info("TRAINING COMPLETED")
        logger.info("========================================")
        
        if valid_dataset is not None:
            logger.info("Running final evaluation...")
            eval_results = trainer.evaluate()
            logger.info(f"Final evaluation results: {eval_results}")
            logger.info("========================================")
            logger.info("EVALUATION COMPLETED")
            logger.info("========================================")
        # Save final model and processor to project_name (like image classification)
        logger.info("========================================")
        logger.info("SAVING MODEL AND PROCESSOR")
        logger.info("========================================")
        
        logger.info("Saving trained model...")
        trainer.save_model(config.project_name)
        
        logger.info("Saving processor...")
        processor.save_pretrained(config.project_name)
        
        # Force-save tokenizer.json if available
        if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "save_pretrained"):
            processor.tokenizer.save_pretrained(config.project_name)
        
        # Create and save model card
        logger.info("Creating model card...")
        model_card = utils.create_asr_model_card(config, trainer)
        with open(f"{config.project_name}/README.md", "w") as f:
            f.write(model_card)
        
        # Remove token from training_config.json before upload
        config_path = os.path.join(config.project_name, "training_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
            if "token" in cfg:
                del cfg["token"]
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)
        # Push model to Hugging Face Hub if push_to_hub is True (main process only)
        if config.push_to_hub:
            if PartialState().process_index == 0:
                logger.info("========================================")
                logger.info("UPLOADING TO HUGGING FACE HUB")
                logger.info("========================================")
                
                remove_autotrain_data(config)

                # Remove token from training_params.json (again, just before upload)
                params_path = os.path.join(config.project_name, "training_params.json")
                if os.path.exists(params_path):
                    with open(params_path, "r") as f:
                        params = json.load(f)
                    if "token" in params:
                        del params["token"]
                    with open(params_path, "w") as f:
                        json.dump(params, f, indent=2)
                    logger.info("Token removed from training_params.json before upload.")

                logger.info("Creating repository...")
                api = HfApi(token=config.token)
                api.create_repo(
                    repo_id=f"{config.username}/{config.project_name}", repo_type="model", private=True, exist_ok=True
                )

                logger.info("Uploading model files...")
                api.upload_folder(
                    folder_path=config.project_name, repo_id=f"{config.username}/{config.project_name}", repo_type="model"
                )
                logger.info(f"Model available at: https://huggingface.co/{config.username}/{config.project_name}")
        if PartialState().process_index == 0:
            logger.info("========================================")
            logger.info("TRAINING PIPELINE COMPLETED")
            logger.info("========================================")
            pause_space(config)
    except Exception as e:
        # Robust error logging if training_logger is not defined
        try:
            logger.error(f"Error in training pipeline: {str(e)}")
        except Exception:
            logger.error(f"Error in training pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, default=None)
    args = parser.parse_args()
    if args.training_config:
        with open(args.training_config, "r", encoding="utf-8") as f:
            config = json.load(f)
        data_path = config.get("data_path", "")
        if data_path == "life_app_data" or os.path.basename(data_path) == "life_app_data":
            from datasets import load_from_disk
            train_dataset = load_from_disk(data_path)
            valid_dataset = None
        else:
            train_dataset = None
            valid_dataset = None

if __name__ == "__main__":
    args = parse_args()
    with open(args.training_config, "r") as f:
        config = json.load(f)
    train(config)