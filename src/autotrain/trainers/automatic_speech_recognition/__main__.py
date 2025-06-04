import argparse
import json
import os
import logging
from typing import Optional, Dict, Any
import torch
import librosa
from datetime import datetime
import pandas as pd
import traceback

from accelerate.state import PartialState
from datasets import load_from_disk, load_dataset, Dataset
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
from transformers.trainer_utils import get_last_checkpoint

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
    DetailedTrainingCallback,
)
from autotrain.trainers.automatic_speech_recognition.dataset import AutomaticSpeechRecognitionDataset
from autotrain.trainers.automatic_speech_recognition.params import AutomaticSpeechRecognitionParams
from autotrain.trainers.automatic_speech_recognition.utils import compute_metrics
from autotrain.logger import get_training_logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()

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
        if params.using_hub_dataset:
            # Load from HuggingFace Hub
            dataset = load_dataset(
                params.data_path,
                split=params.valid_split if is_validation else params.train_split,
                use_auth_token=params.token if params.token else None
            )
        else:
            # Load from local directory
            logger.info(f"Looking for data in: {params.data_path}")
            logger.info(f"Current working directory: {os.getcwd()}")
            
            # Check if data_path exists
            if not os.path.exists(params.data_path):
                logger.error(f"Data path does not exist: {params.data_path}")
                raise ValueError(f"Data path does not exist: {params.data_path}")
            
            # List all files in the directory
            files = os.listdir(params.data_path)
            logger.info(f"Files in directory: {files}")
            
            # Look for dataset.csv
            if 'dataset.csv' in files:
                logger.info("Found dataset.csv, loading...")
                csv_file_path = os.path.join(params.data_path, 'dataset.csv')
            else:
                # Look for any CSV file
                csv_files = [f for f in files if f.endswith('.csv')]
                if csv_files:
                    csv_file_path = os.path.join(params.data_path, csv_files[0])
                    logger.info(f"Using CSV file: {csv_files[0]}")
                else:
                    raise ValueError(f"No CSV file found in {params.data_path}")
            
            # Read CSV file using pandas
            logger.info(f"Loading CSV file: {csv_file_path}")
            df = pd.read_csv(csv_file_path)
            
            # Log dataset info
            logger.info(f"CSV columns: {df.columns.tolist()}")
            logger.info(f"Number of examples: {len(df)}")
            
            # Map column names if needed
            if 'autotrain_audio' in df.columns:
                logger.info("Mapping autotrain_audio to audio")
                df = df.rename(columns={'autotrain_audio': 'audio'})
            if 'autotrain_transcription' in df.columns:
                logger.info("Mapping autotrain_transcription to transcription")
                df = df.rename(columns={'autotrain_transcription': 'transcription'})
            
            # Verify required columns
            required_columns = ['audio', 'transcription']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in CSV: {missing_columns}. Available columns: {df.columns.tolist()}")
            
            # Split into train/valid if needed
            if is_validation:
                # Use 20% of data for validation
                valid_size = int(len(df) * 0.2)
                df = df.sample(n=valid_size, random_state=42)
                logger.info(f"Using {valid_size} examples for validation")
            else:
                # Use 80% of data for training
                train_size = int(len(df) * 0.8)
                df = df.sample(n=train_size, random_state=42)
                logger.info(f"Using {train_size} examples for training")
            
            # Verify audio files exist and update paths if needed
            invalid_audio_files = []
            audio_folder = os.path.join(params.data_path, 'audio')
            
            # Check if audio folder exists
            if not os.path.exists(audio_folder):
                logger.warning(f"Audio folder not found: {audio_folder}")
                logger.warning("Will try to use audio paths as provided in CSV")
            
            for idx, audio_path in enumerate(df['audio']):
                # Convert to absolute path if relative
                if not os.path.isabs(audio_path):
                    # Try to find audio file in audio folder
                    if os.path.exists(audio_folder):
                        audio_file = os.path.join(audio_folder, os.path.basename(audio_path))
                        if os.path.exists(audio_file):
                            df.at[idx, 'audio'] = os.path.abspath(audio_file)
                            continue
                    # If not found in audio folder, try relative to data_path
                    audio_file = os.path.join(params.data_path, audio_path)
                    if os.path.exists(audio_file):
                        df.at[idx, 'audio'] = os.path.abspath(audio_file)
                        continue
                
                # If absolute path, check if it exists
                if not os.path.exists(audio_path):
                    invalid_audio_files.append((idx, audio_path))
            
            if invalid_audio_files:
                error_msg = "The following audio files were not found:\n"
                for idx, path in invalid_audio_files[:5]:  # Show first 5 errors
                    error_msg += f"Row {idx}: {path}\n"
                if len(invalid_audio_files) > 5:
                    error_msg += f"... and {len(invalid_audio_files) - 5} more files"
                raise ValueError(error_msg)
            
            # Convert pandas DataFrame to Datasets Dataset
            dataset = Dataset.from_pandas(df)
            
            # Log first example for verification
            if len(dataset) > 0:
                logger.info("First example in dataset:")
                logger.info(f"Audio path: {dataset[0]['audio']}")
                logger.info(f"Transcription: {dataset[0]['transcription']}")
                if 'duration' in dataset[0]:
                    logger.info(f"Duration: {dataset[0]['duration']}")
                
                # Verify audio file can be loaded
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
        # Load processor
        logger.info(f"Loading processor: {params.model}")
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
        
        # Determine model type and load appropriate model
        model_name = params.model.lower()
        
        # Try loading as Seq2Seq model first (for Whisper, MMS, etc.)
        try:
            logger.info("Attempting to load as Seq2Seq model...")
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                params.model,
                token=params.token if params.token else None,
                trust_remote_code=ALLOW_REMOTE_CODE,
            )
            logger.info("Successfully loaded as Seq2Seq model")
        except Exception as e:
            logger.info(f"Could not load as Seq2Seq model: {str(e)}")
            # Try loading as CTC model (for Wav2Vec2, Hubert, etc.)
            try:
                logger.info("Attempting to load as CTC model...")
                model = AutoModelForCTC.from_pretrained(
                    params.model,
                    token=params.token if params.token else None,
                    trust_remote_code=ALLOW_REMOTE_CODE,
                )
                logger.info("Successfully loaded as CTC model")
            except Exception as e:
                logger.info(f"Could not load as CTC model: {str(e)}")
                # Try loading as generic model
                try:
                    logger.info("Attempting to load as generic model...")
                    model = AutoModel.from_pretrained(
                        params.model,
                        token=params.token if params.token else None,
                        trust_remote_code=ALLOW_REMOTE_CODE,
                    )
                    logger.info("Successfully loaded as generic model")
                except Exception as e:
                    raise ValueError(f"Could not load model {params.model} as any supported type: {str(e)}")
        
        # Move model to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # Log model configuration
        logger.info(f"Model type: {model.__class__.__name__}")
        logger.info(f"Model configuration: {model.config}")
        
        # Check if model has required methods
        required_methods = ['forward', 'get_encoder', 'get_decoder']
        missing_methods = [method for method in required_methods if not hasattr(model, method)]
        if missing_methods:
            logger.warning(f"Model is missing some required methods: {missing_methods}")
            logger.warning("This might affect training. Please check model documentation.")
        
        # Check model parameters
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
    """
    Train ASR model.
    
    Args:
        config: Training configuration
    """
    try:
        # Parse parameters
        params = AutomaticSpeechRecognitionParams(**config)

        # Setup detailed logging
        training_logger = get_training_logger(params.output_dir)
        training_logger.info("Initializing ASR training...")

        # Fix data path if needed
        if not params.using_hub_dataset:
            # Set the correct data path
            data_path = r"C:\Users\Aryan\Downloads\archive"
            training_logger.info(f"Using fixed data path: {data_path}")

            # Verify paths exist
            csv_path = os.path.join(data_path, "dataset.csv")
            audio_path = os.path.join(data_path, "audio")

            if not os.path.exists(csv_path):
                raise ValueError(f"CSV file not found at: {csv_path}")
            if not os.path.exists(audio_path):
                raise ValueError(f"Audio folder not found at: {audio_path}")

            # Update data path
            params.data_path = data_path
            training_logger.info(f"CSV file path: {csv_path}")
            training_logger.info(f"Audio folder path: {audio_path}")

        # Validate parameters
        params.validate_params()

        # Log parameters
        training_logger.info("Training parameters:")
        training_logger.info(json.dumps(params.model_dump(), indent=2))

        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        training_logger.info(f"Using device: {device}")
        training_logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            training_logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            training_logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            training_logger.info(f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

        # Load dataset
        training_logger.info("Loading dataset...")
        training_logger.info(f"Dataset path: {params.data_path}")
        training_logger.info(f"Using hub dataset: {params.using_hub_dataset}")

        if params.using_hub_dataset:
            # Load from HuggingFace Hub
            training_logger.info("Loading dataset from HuggingFace Hub...")
            dataset = load_data(params)
            training_logger.info(f"Successfully loaded {len(dataset)} examples from hub")
        else:
            # Load from local directory
            training_logger.info("Loading dataset from local directory...")
            dataset = load_data(params)
            training_logger.info(f"Successfully loaded {len(dataset)} examples from local directory")

            # Verify dataset structure
            if len(dataset) == 0:
                raise ValueError("Dataset is empty")

            # Log first few examples
            training_logger.info("First few examples in dataset:")
            for i in range(min(3, len(dataset))):
                training_logger.info(f"Example {i}:")
                training_logger.info(f"  Audio path: {dataset[i]['audio']}")
                training_logger.info(f"  Transcription: {dataset[i]['transcription']}")

        # Load validation dataset if available
        if params.valid_split:
            training_logger.info(f"Loading validation dataset: {params.valid_split}")
            valid_dataset = load_data(params, is_validation=True)
            training_logger.info(f"Successfully loaded {len(valid_dataset)} validation examples")
        else:
            training_logger.info("No validation file found")
            valid_dataset = None

        # Load model and processor
        training_logger.info(f"Loading model and processor: {params.model}")
        model, processor = load_model_and_processor(params)

        # Log model info
        training_logger.info("Successfully loaded model: " + model.__class__.__name__)
        training_logger.info("Successfully loaded processor: " + processor.__class__.__name__)

        # Log model configuration
        training_logger.info("Model configuration:")
        training_logger.info(f"Model type: {model.__class__.__name__}")
        training_logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
        training_logger.info(f"Model device: {next(model.parameters()).device}")

        # Create training dataset
        training_logger.info("Creating training dataset...")
        training_logger.info(f"Dataset columns before creation: {dataset.column_names}")
        if len(dataset) > 0:
            training_logger.info(f"First example before creation: {dataset[0]}")

        try:
            train_dataset = AutomaticSpeechRecognitionDataset(
                data=dataset,
                processor=processor,
                config=params,
                audio_column=params.audio_column,
                text_column=params.text_column,
                max_duration=params.max_duration,
                sampling_rate=params.sampling_rate,
            )
            training_logger.info(f"Successfully created training dataset with {len(train_dataset)} examples")
            # Log first example
            training_logger.info(f"First example after creation: {train_dataset[0]}")
        except Exception as e:
            training_logger.error(f"Error creating dataset: {str(e)}")
            raise

        # Create validation dataset if available
        if valid_dataset is not None:
            training_logger.info("Creating validation dataset...")
            try:
                valid_dataset = AutomaticSpeechRecognitionDataset(
                    data=valid_dataset,
                    processor=processor,
                    config=params,
                    audio_column=params.audio_column,
                    text_column=params.text_column,
                    max_duration=params.max_duration,
                    sampling_rate=params.sampling_rate,
                )
                training_logger.info(f"Successfully created validation dataset with {len(valid_dataset)} examples")
            except Exception as e:
                training_logger.error(f"Error creating validation dataset: {str(e)}")
                raise

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=params.output_dir,
            per_device_train_batch_size=params.batch_size,
            per_device_eval_batch_size=params.batch_size,
            gradient_accumulation_steps=params.gradient_accumulation,
            learning_rate=params.lr,
            num_train_epochs=params.epochs,
            save_strategy="epoch",
            evaluation_strategy="epoch" if valid_dataset else "no",
            load_best_model_at_end=True if valid_dataset else False,
            metric_for_best_model="wer" if valid_dataset else None,
            greater_is_better=False if valid_dataset else None,
            push_to_hub=params.push_to_hub,
            hub_model_id=params.hub_model_id,
            hub_token=params.token,
            logging_dir=os.path.join(params.output_dir, "logs"),
            logging_steps=10,
            save_total_limit=2,
            remove_unused_columns=False,
            fp16=params.mixed_precision == "fp16",
            bf16=params.mixed_precision == "bf16",
            # Add more robust training arguments
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            gradient_checkpointing=True,
            optim="adamw_torch",
            lr_scheduler_type="linear",
            warmup_ratio=0.1,
            weight_decay=0.01,
            max_grad_norm=1.0,
            report_to="tensorboard",
            seed=42,
        )

        # Create trainer with detailed logging callback
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            tokenizer=processor,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                LossLoggingCallback(),
                TrainStartCallback(),
                UploadLogs(),
                DetailedTrainingCallback(),
            ],
        )

        # Train model
        training_logger.info("Starting training...")
        trainer.train()
        
        # Save model
        training_logger.info("Saving model...")
        trainer.save_model(params.output_dir)
        processor.save_pretrained(params.output_dir)

        # Create model card
        training_logger.info("Creating model card...")
        model_card = create_model_card(params, trainer)
        with open(os.path.join(params.output_dir, "README.md"), "w") as f:
            f.write(model_card)

        # Push to hub if requested
        if params.push_to_hub:
            training_logger.info("Pushing model to hub...")
            trainer.push_to_hub()

        training_logger.info("Training completed successfully!")

    except Exception as e:
        training_logger.error(f"Error during training: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main():
    """Main function to run ASR training."""
    try:
        # Load training config
        with open("training_config.json", "r") as f:
            training_config = json.load(f)
            
        logger.info("Initializing ASR training...")
        
        # Initialize trainer with detailed logging
        trainer = ASRTrainer(
            training_config=training_config,
            callbacks=[DetailedTrainingCallback()]  # Add detailed logging callback
        )
        
        # Start training with detailed progress
        logger.info("Starting ASR training with detailed progress logging...")
        trainer.train()
        
    except Exception as e:
        logger.error(f"Error during ASR training: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

if __name__ == "__main__":
    args = parse_args()
    with open(args.training_config, "r") as f:
        config = json.load(f)
    train(config) 