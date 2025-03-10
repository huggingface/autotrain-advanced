import numpy as np
import pytest
from datasets import Dataset
from transformers import WhisperProcessor

from autotrain.trainers.asr import WhisperTrainingParams
from autotrain.trainers.asr.utils import load_audio_dataset, prepare_dataset, compute_metrics


def create_dummy_dataset():
    """Create a dummy dataset for testing."""
    return Dataset.from_dict({
        "audio": [
            {
                "array": np.zeros(16000),  # 1 second of silence
                "sampling_rate": 16000
            }
        ],
        "text": ["test transcription"]
    })


def test_whisper_training_params():
    """Test WhisperTrainingParams initialization and LoRA config."""
    params = WhisperTrainingParams()
    
    # Test default values
    assert params.sampling_rate == 16000
    assert params.max_duration_secs == 30.0
    assert params.use_peft is True
    
    # Test LoRA config
    lora_config = params.get_lora_config()
    assert lora_config is not None
    assert lora_config.r == params.lora_r
    assert lora_config.lora_alpha == params.lora_alpha
    
    # Test PEFT disabled
    params.use_peft = False
    assert params.get_lora_config() is None


def test_load_audio_dataset():
    """Test dataset loading and validation."""
    dataset = create_dummy_dataset()
    
    # Test successful loading
    loaded_dataset = load_audio_dataset(
        dataset_path=dataset,
        audio_column="audio",
        text_column="text"
    )
    assert isinstance(loaded_dataset, Dataset)
    
    # Test missing columns
    with pytest.raises(ValueError):
        load_audio_dataset(
            dataset_path=dataset,
            audio_column="nonexistent",
            text_column="text"
        )


def test_prepare_dataset(mocker):
    """Test dataset preparation and processing."""
    dataset = create_dummy_dataset()
    processor = mocker.Mock(spec=WhisperProcessor)
    
    # Mock processor methods
    processor.return_value = {"input_features": np.zeros((1, 80, 3000))}
    processor.input_ids = [1, 2, 3]
    
    # Test dataset preparation
    processed_dataset = prepare_dataset(
        dataset=dataset,
        processor=processor,
        max_duration_secs=1.0
    )
    
    assert "input_features" in processed_dataset.features
    assert "labels" in processed_dataset.features


def test_compute_metrics(mocker):
    """Test WER metric computation."""
    # Mock predictions and labels
    predictions = np.array([[1, 2, 3]])
    labels = np.array([[1, 2, 3]])
    
    # Mock processor
    processor = mocker.Mock()
    processor.batch_decode.side_effect = [
        ["hello world"],  # predictions
        ["hello world"]   # references
    ]
    
    # Create prediction object
    class Predictions:
        def __init__(self):
            self.predictions = predictions
            self.label_ids = labels
    
    pred = Predictions()
    
    # Test metric computation
    metrics = compute_metrics(pred)
    assert "wer" in metrics
    assert isinstance(metrics["wer"], float) 