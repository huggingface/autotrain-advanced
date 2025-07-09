import os
import torch
import librosa
from torch.utils.data import Dataset
from autotrain import logger

# Figure out what kind of ASR model we're working with

def detect_model_type(model):
    model_class = type(model).__name__
    if 'Whisper' in model_class or 'Seq2Seq' in model_class:
        return "seq2seq"
    elif 'CTC' in model_class or 'Wav2Vec' in model_class or 'Hubert' in model_class:
        return "ctc"
    else:
        return "generic"

# Tokenize text safely, handling different processor/tokenizer types

def safe_tokenize_text(processor, text, max_seq_length=448):
    if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'encode'):
        try:
            input_ids = processor.tokenizer.encode(
                text,
                max_length=max_seq_length,
                truncation=True,
                return_tensors="pt"
            )
            return input_ids.squeeze(0)
        except Exception:
            pass
    if hasattr(processor, "tokenizer"):
        try:
            return processor.tokenizer(
                text,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.squeeze(0)
        except Exception:
            pass
    try:
        return processor(
            text,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)
    except Exception:
        pass
    # Fallback: just return a tensor of zeros
    return torch.tensor([0] * min(max_seq_length, 10), dtype=torch.long)

class AutoTrainASRDataset(Dataset):
    """
    Universal ASR Dataset for all model types (Whisper, Wav2Vec2, Hubert, etc.).
    Handles model type detection, tokenization, audio loading, and error fallback.
    """

    def __init__(
        self,
        data,
        processor,
        model,
        audio_column="audio",
        text_column="transcription",
        max_duration=30.0,
        sampling_rate=16000,
        model_type=None,
        max_seq_length=448,
    ):
        self._data = data
        self.processor = processor
        self.model = model
        self.audio_column = audio_column
        self.text_column = text_column
        self.max_duration = max_duration
        self.sampling_rate = sampling_rate
        self.max_seq_length = max_seq_length
        self.model_type = model_type or detect_model_type(model)
        # Check processor
        if not hasattr(self, 'processor') or self.processor is None:
            logger.warning("Processor is NOT set on ASR dataset! Labels will be broken.")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        try:
            item = self._data[idx]
            audio_path = item[self.audio_column]
            text = item[self.text_column] if self.text_column in item else None
            # Encode the text using the processor (batch call, like seq2seq)
            label_ids = None
            decoded_from_ids = None
            if text and isinstance(text, str) and hasattr(self, 'processor') and self.processor is not None:
                try:
                    label_ids_tensor = self.processor.tokenizer(
                        text,
                        max_length=self.max_seq_length,
                        truncation=True,
                        return_tensors="pt",
                        add_special_tokens=True,
                    ).input_ids.squeeze(0)
                    label_ids = label_ids_tensor.tolist()
                    decoded_from_ids = self.processor.tokenizer.decode(label_ids)
                except Exception as e:
                    logger.warning(f"Exception in encoding/decoding: {e}")
                    raise ValueError(f"Failed to encode text '{text}' at idx {idx}: {e}")
            else:
                raise ValueError(f"Text column '{self.text_column}' is missing or invalid in item at idx {idx}!")
            if not os.path.exists(audio_path):
                raise ValueError(f"Audio file not found: {audio_path}")
            # Load the audio file and check its duration
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
            duration = len(audio) / sr
            if duration > self.max_duration:
                max_samples = int(self.max_duration * self.sampling_rate)
                logger.info(f"Truncating file {audio_path} from {duration:.2f}s to {self.max_duration}s")
                audio = audio[:max_samples]
            # Prepare model input depending on model type
            if self.model_type == 'seq2seq':
                try:
                    inputs = self.processor(
                        audio,
                        sampling_rate=self.sampling_rate,
                        return_tensors="pt",
                        padding=False,
                        truncation=True,
                    )
                    input_features = inputs.input_features[0]
                    target_length = 3000
                    # Pad or truncate to a fixed length for seq2seq models
                    if input_features.shape[1] < target_length:
                        padding = torch.zeros(80, target_length - input_features.shape[1])
                        input_features = torch.cat([input_features, padding], dim=1)
                    elif input_features.shape[1] > target_length:
                        input_features = input_features[:, :target_length]
                except Exception:
                    input_features = torch.tensor(audio, dtype=torch.float32)
            elif self.model_type == 'ctc':
                try:
                    inputs = self.processor(
                        audio,
                        sampling_rate=self.sampling_rate,
                        return_tensors="pt",
                        padding=False,
                        truncation=True,
                    )
                    input_features = inputs.input_values[0]
                except Exception:
                    input_features = torch.tensor(audio, dtype=torch.float32)
            else:
                try:
                    inputs = self.processor(
                        audio,
                        sampling_rate=self.sampling_rate,
                        return_tensors="pt",
                        padding=False,
                        truncation=True,
                    )
                    if hasattr(inputs, 'input_features'):
                        input_features = inputs.input_features[0]
                    elif hasattr(inputs, 'input_values'):
                        input_features = inputs.input_values[0]
                    else:
                        input_features = torch.tensor(audio, dtype=torch.float32)
                except Exception:
                    input_features = torch.tensor(audio, dtype=torch.float32)
            labels = torch.tensor(label_ids, dtype=torch.long)

            if self.model_type == 'seq2seq':
                return {
                    "input_features": input_features,
                    "labels": labels,
                }
            else:
                return {
                    "input_values": input_features,
                    "labels": labels,
                }
        except Exception:
            # If anything goes wrong, return dummy data so training doesn't crash
            if self.model_type == 'seq2seq':
                dummy_audio = torch.zeros(80, 100, dtype=torch.float32)
            else:
                dummy_audio = torch.zeros(1000, dtype=torch.float32)
            dummy_labels = torch.tensor([0], dtype=torch.long)
            if self.model_type == 'seq2seq':
                return {
                    "input_features": dummy_audio,
                    "labels": dummy_labels,
                }
            else:
                return {
                    "input_values": dummy_audio,
                    "labels": dummy_labels,
                }

# Load a LiFE App dataset from disk and split into train/validation

def load_life_app_dataset(data_path):
    """
    Loads the LiFE App dataset from disk for ASR training.
    """
    import pandas as pd
    from datasets import Dataset, DatasetDict
    import os

    logger.info(f"ASR pipeline: loading dataset from {data_path}")
    csv_path = os.path.join(data_path, "processed_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"LiFE App processed CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    # Split 80/20 for train/validation
    split_idx = int(0.8 * len(df))
    train_df = df.iloc[:split_idx]
    valid_df = df.iloc[split_idx:]
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    dataset = DatasetDict({"train": train_dataset, "validation": valid_dataset})
    logger.info(f"ASR pipeline: loaded dataset with {len(dataset['train'])} train and {len(dataset['validation'])} validation samples")
    return dataset