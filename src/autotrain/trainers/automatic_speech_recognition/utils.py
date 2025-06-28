import json
from typing import Dict, Any
import numpy as np
from transformers import Trainer
import jiwer


processor = None

def set_processor(proc):
    global processor
    processor = proc

def compute_metrics(pred):
    import jiwer
    import numpy as np

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    if isinstance(label_ids, tuple):
        label_ids = label_ids[0]

    
    pred_ids = np.asarray(pred_ids)
    label_ids = np.asarray(label_ids)

   
    if pred_ids.dtype != np.int32 and pred_ids.dtype != np.int64:
        pred_ids = np.argmax(pred_ids, axis=-1)

    
    label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)

    
    pred_ids = pred_ids.astype(int).tolist()
    label_ids = label_ids.astype(int).tolist()

    
    if isinstance(pred_ids[0], list):
        pass  
    else:
        pred_ids = [pred_ids]
    if isinstance(label_ids[0], list):
        pass
    else:
        label_ids = [label_ids]

    
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    
    wer = jiwer.wer(label_str, pred_str)
    cer = jiwer.cer(label_str, pred_str)

    return {"wer": wer, "cer": cer}

def create_model_card(config: Dict[str, Any], trainer: Trainer, num_classes: int = None) -> str:
    """Create a model card for the trained ASR model."""
    model_card = f"""# {config.project_name}

This model was trained using AutoTrain.

## Training Configuration

- Base Model: {config.model}
- Task: Automatic Speech Recognition
- Training Data: {config.data_path}
- Validation Data: {config.valid_split if config.valid_split else "None"}
- Epochs: {config.epochs}
- Batch Size: {config.batch_size}
- Learning Rate: {config.lr}
- Optimizer: {config.optimizer}
- Scheduler: {config.scheduler}
- Mixed Precision: {config.mixed_precision}

## Training Results

- Final Loss: {next((entry['loss'] for entry in reversed(trainer.state.log_history) if 'loss' in entry), "N/A")}
- Best Validation Loss: {trainer.state.best_metric if hasattr(trainer.state, 'best_metric') else "N/A"}

## Usage

```python
from transformers import AutoModelForCTC, Wav2Vec2Processor
import torch
import librosa

# Load model and processor
model = AutoModelForCTC.from_pretrained("{config.username}/{config.project_name}")
processor = Wav2Vec2Processor.from_pretrained("{config.username}/{config.project_name}")

# Load and preprocess audio
audio, sr = librosa.load("path_to_audio.wav", sr={config.sampling_rate})
inputs = processor(audio, sampling_rate={config.sampling_rate}, return_tensors="pt", padding=True)

# Get predictions
with torch.no_grad():
    logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
```
"""
    return model_card 