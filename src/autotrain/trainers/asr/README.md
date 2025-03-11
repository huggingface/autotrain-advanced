# AutoTrain ASR Module

This module provides functionality for fine-tuning Whisper models for Automatic Speech Recognition (ASR) using PEFT/LoRA for efficient training.

## Features

- Memory-efficient fine-tuning using PEFT/LoRA
- Support for custom audio preprocessing parameters
- Flexible dataset column mapping
- Progress tracking with Word Error Rate (WER) metric
- Proper model and adapter saving

## Usage

### Using Configuration File

Create a YAML configuration file:

```yaml
# ASR Training Configuration
model_name: openai/whisper-small
dataset_path: your-dataset-name-or-path
output_dir: whisper-finetuned
audio_column: audio
text_column: text

# Audio processing parameters
sampling_rate: 16000
max_duration_secs: 30.0
preprocessing_num_workers: 4

# Model parameters
language: en
task: transcribe

# Training parameters
learning_rate: 5e-5
num_train_epochs: 3
per_device_train_batch_size: 8
gradient_accumulation_steps: 1

# PEFT/LoRA parameters
use_peft: true
lora_r: 8
lora_alpha: 32
lora_dropout: 0.1
```

Then run:

```bash
python -m autotrain.trainers.asr --training_config path/to/config.yaml
```

### Using Python API

```python
from autotrain.trainers.asr import train_whisper, WhisperTrainingParams

# Configure training parameters
params = WhisperTrainingParams(
    model_name="openai/whisper-small",
    language="en",
    use_peft=True,
    lora_r=8,
    lora_alpha=32
)

# Start training
train_whisper(
    params=params,
    dataset_path="your-dataset-name-or-path",  # HF dataset name or local path
    output_dir="whisper-finetuned",
    audio_column="audio",
    text_column="text"
)
```

## Requirements

- transformers>=4.48.0
- datasets>=3.2.0
- peft>=0.14.0
- librosa>=0.10.1
- soundfile>=0.12.1
- jiwer>=3.0.5

## Dataset Format

Your dataset should contain at least two columns:
1. Audio column: Contains audio data in the format expected by the datasets library
2. Text column: Contains the transcription text

Example dataset format:
```python
{
    'audio': {
        'array': np.array(...),  # Audio samples
        'sampling_rate': 16000   # Original sampling rate
    },
    'text': 'transcription of the audio'
}
```

## Advanced Configuration

See `WhisperTrainingParams` class for all available configuration options:
- Audio processing parameters
- Model configuration
- Training hyperparameters
- PEFT/LoRA settings 