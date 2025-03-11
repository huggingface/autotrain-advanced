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


def test_load_audio_dataset(mocker):
    """Test dataset loading and validation."""
    dataset = create_dummy_dataset()
    
    # Mock load_dataset to return our dummy dataset
    mocker.patch('autotrain.trainers.asr.utils.load_dataset', return_value=dataset)
    
    # Test successful loading
    loaded_dataset = load_audio_dataset(
        dataset_path="dummy_path",
        audio_column="audio",
        text_column="text"
    )
    assert isinstance(loaded_dataset, Dataset)
    
    # Test missing columns
    with pytest.raises(ValueError):
        # Create dataset with missing column
        bad_dataset = Dataset.from_dict({"text": ["test"]})
        mocker.patch('autotrain.trainers.asr.utils.load_dataset', return_value=bad_dataset)
        load_audio_dataset(
            dataset_path="dummy_path",
            audio_column="audio",
            text_column="text"
        )


def test_prepare_dataset(mocker):
    """Test dataset preparation and processing."""
    dataset = create_dummy_dataset()
    
    # Create a proper mock for the processor
    processor = mocker.Mock(spec=WhisperProcessor)
    
    # Create a mock for the return value of processor.__call__
    mock_features = mocker.Mock()
    mock_features.input_features = [np.zeros((80, 3000))]
    
    # Set up the processor to return the mock_features when called
    processor.side_effect = lambda *args, **kwargs: mock_features
    
    # Mock the processor's tokenizer call for text processing
    processor.input_ids = [1, 2, 3]
    
    # Test dataset preparation with mocked map function to avoid actual processing
    mocker.patch.object(dataset, 'map', return_value=Dataset.from_dict({
        "input_features": [np.zeros((80, 3000))],
        "labels": [[1, 2, 3]]
    }))
    
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
    processor.tokenizer = mocker.Mock()
    processor.tokenizer.pad_token_id = 0
    processor.batch_decode.side_effect = [
        ["hello world"],  # predictions
        ["hello world"]   # references
    ]
    
    # Mock evaluate module
    evaluate_mock = mocker.Mock()
    wer_mock = mocker.Mock()
    wer_mock.compute.return_value = 0.0
    evaluate_mock.load.return_value = wer_mock
    
    # Patch the evaluate module at the module level
    mocker.patch('autotrain.trainers.asr.utils.evaluate', evaluate_mock)
    
    # Create prediction object
    class Predictions:
        def __init__(self):
            self.predictions = predictions
            self.label_ids = labels
    
    pred = Predictions()
    
    # Test metric computation
    metrics = compute_metrics(pred, processor)
    assert "wer" in metrics 