# AutoTrain Advanced: Automatic Speech Recognition (ASR) Task

This guide explains how to use AutoTrain Advanced for **Automatic Speech Recognition (ASR)** tasks, including supported models, data formats, configuration, training, evaluation metrics (WER, CER), and advanced usage. It also documents improvements and fixes.

---

## Overview
- Train with local datasets (audio + CSV/JSON)
- Train with Hugging Face Hub datasets
- Seamless integration with LiFE App datasets
- Consistent workflow, UI, and code hygiene as other tasks (image, text, etc.)

---

## Supported ASR Models
- Any Hugging Face model compatible with `AutoModelForCTC` or `AutoModelForSpeechSeq2Seq`
- Example: `facebook/wav2vec2-base-960h`, `openai/whisper-base`, etc.

---

## File Structure
```
src/autotrain/trainers/automatic_speech_recognition/
    __main__.py         # Training entry point and pipeline
    dataset.py          # Universal ASR dataset class
    utils.py            # Metrics, model card, callbacks
    params.py           # Training parameter config
src/autotrain/preprocessor/automatic_speech_recognition.py  # Local/LiFE App dataset validation/prep
```

---

## Data Format
Your dataset must have:
- **audio**: Path to audio file (WAV/FLAC/MP3, etc.)
- **transcription**: The ground truth text for the audio

### Example CSV
| audio           | transcription      |
|-----------------|-------------------|
| audio_0.wav     | hello world       |
| audio_1.wav     | how are you       |

- Place all audio files in a folder (e.g., `audio/`).
- The CSV/JSON file should reference the correct relative or absolute path to each audio file.

---

## Supported Dataset Sources
- **Local**: Upload a zip with an audio folder and a CSV/JSON file (columns: `audio`, `transcription`).
- **Hugging Face Hub**: Use a public dataset (e.g., `mozilla-foundation/common_voice_11_0`).
- **LiFE App**: (ASR only) Select a LiFE App project/script; backend converts to local format automatically.

### Local Dataset Format
- Folder must contain:
  - `audio/` directory with audio files (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`)
  - CSV or JSON file with columns:
    - `audio`: filename (not path)
    - `transcription`: ground truth text

### LiFE App Integration
- LiFE App datasets are converted to the above local format automatically.
- All validation, mapping, and error handling is shared with other sources.

---

## Example Config File (YAML)
```yaml
# Example: configs/automatic_speech_recognition/local_dataset.yml
task: ASR
base_model: facebook/wav2vec2-base-960h
project_name: autotrain-asr-demo
log: tensorboard
backend: local

data:
  path: data/                # Folder with your CSV/JSON and audio/
  train_split: train         # Name of your training split (e.g., train.csv)
  valid_split: valid         # Name of your validation split (e.g., valid.csv)
  column_mapping:
    audio_column: audio
    text_column: transcription

params:
  max_duration: 30.0
  sampling_rate: 16000
  max_seq_length: 128
  epochs: 3
  batch_size: 8
  lr: 3e-4
  optimizer: adamw_torch
  scheduler: linear
  gradient_accumulation: 1
  mixed_precision: fp16
  weight_decay: 0.01
  warmup_ratio: 0.1
  early_stopping_patience: 3
  early_stopping_threshold: 0.01

hub:
  username: ${HF_USERNAME}
  token: ${HF_TOKEN}
  push_to_hub: false
```

---

## Usage
### 1. From the UI
- Select **ASR** as the task.
- Choose dataset source (Local, Hub, or LiFE App).
- Map columns as needed.
- Set parameters (basic or full).
- Start training. Progress and logs are shown in the UI.

### 2. From the CLI
```bash
autotrain --config configs/automatic_speech_recognition/local_dataset.yml
```
Or:
```bash
python -m autotrain.trainers.automatic_speech_recognition --training_config path/to/config.json
```

---

## Training Parameters
All standard parameters are supported (see `params.py`):
- `model`: Pretrained model name or path (e.g., `facebook/wav2vec2-base-960h`)
- `batch_size`, `epochs`, `lr`, `warmup_ratio`, `weight_decay`, etc.
- `audio_column`, `text_column`, `max_duration`, `sampling_rate`, `max_seq_length`
- `push_to_hub`, `logging_steps`, `early_stopping_patience`, etc.

---

## Evaluation Metrics
During validation, AutoTrain computes:
- **WER (Word Error Rate):** Measures word-level errors between prediction and ground truth.
- **CER (Character Error Rate):** Measures character-level errors.
- **Accuracy:** Exact match between predicted and ground truth text.

**How are they computed?**
- The model's predicted text and the ground truth transcription are compared for each sample.
- WER and CER are calculated using standard Levenshtein distance.

**Example Output:**
```
Validation Results:
  WER: 0.12
  CER: 0.05
  Accuracy: 0.88
```

**Model Card Snippet:**
```
## Validation Metrics
wer: 0.12

cer: 0.05

accuracy: 0.88
```

---

## Tips for Best Results
- Ensure all audio files are present and paths are correct in your CSV/JSON.
- Transcriptions should be clean and accurate.
- Use a sampling rate compatible with your model (e.g., 16kHz for wav2vec2).
- For large datasets, use the Hugging Face Hub for faster loading.

---

## Advanced Usage
- **Custom Models:**
  - Add new model support by updating `detect_model_type` in `dataset.py` and model loading logic in `__main__.py`.
- **Custom Metrics:**
  - Extend or replace `compute_metrics` in `utils.py`.
- **Custom Callbacks:**
  - Add new callbacks in `utils.py` and register them in the `train` function in `__main__.py`.

---

## Example Output
**Training Log Snippet:**
```
[INFO] Using device: cuda
[INFO] Training dataset object created with 1200 examples.
[INFO] TRAINING STARTED - Watch for progress logs
[INFO] TRAINING COMPLETED
[INFO] Final evaluation results: {'eval_loss': 0.32, 'eval_wer': 0.12, 'eval_cer': 0.05, 'eval_accuracy': 0.88}
```

---

## ASR Training Pipeline (Diagram)
```mermaid
graph TD;
  A[Start: Select ASR Task] --> B{Choose Dataset Source};
  B -->|Local| C[Upload audio + CSV/JSON];
  B -->|Hugging Face Hub| D[Enter dataset name/split];
  B -->|LiFE App| E[Select project/script];
  C & D & E --> F[Preprocessing & Validation];
  F --> G[Model/Processor Loading];
  G --> H[Training Loop];
  H --> I[Evaluation & Metrics];
  I --> J[Model Card & Save];
  J --> K[Push to Hub (optional)];
```

---

## Troubleshooting
- **Missing audio files:** Check that all paths in your CSV/JSON are correct.
- **High WER/CER:** Check your transcriptions for typos, and ensure audio quality is good.
- **Parameter errors:** Make sure your config matches the expected format.

---

## Further Reading & Support
- [AutoTrain Advanced Main Repo](https://github.com/huggingface/autotrain-advanced)
- [AutoTrain Advanced Documentation](https://huggingface.co/docs/autotrain)
- [Sample ASR Configs](https://github.com/huggingface/autotrain-advanced/tree/main/configs/automatic_speech_recognition)
- [Supported Models](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition)
- [Open an Issue](https://github.com/huggingface/autotrain-advanced/issues)

---

## Work Done / Improvements
- Fixed the previous issue where users would see repeated or false 'token verification failed' errors. Authentication logic is now robust and user-friendly, ensuring smooth login and operation for both UI and backend.

---

**Happy Training!** 