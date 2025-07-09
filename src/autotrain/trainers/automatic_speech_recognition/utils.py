import os
import numpy as np
from jiwer import wer, cer
from transformers import TrainerCallback
from autotrain.trainers.automatic_speech_recognition.dataset import AutoTrainASRDataset
from autotrain import logger
import jiwer
# Metrics tracked for ASR evaluation
ASR_EVAL_METRICS = (
    "eval_loss",
    "eval_wer",
    "eval_cer",
    "eval_accuracy",
)

# Template for ASR model card
ASR_MODEL_CARD = """
---
tags:
- autotrain
- transformers
- automatic-speech-recognition{base_model}
widget:
- example_title: Example Audio
  src: <audio file url>
{dataset_tag}
---

# Model Trained Using AutoTrain

- Problem type: Automatic Speech Recognition

## Validation Metrics
{validation_metrics}
"""

def _asr_metrics(pred):
    """
    Compute WER, CER, and accuracy for ASR predictions.
    Args:
        pred (tuple): (predictions, labels) as lists of strings.
    Returns:
        dict: WER, CER, accuracy.
    """
    predictions, labels = pred
    wer_score = wer(labels, predictions)
    cer_score = cer(labels, predictions)
    accuracy = np.mean([p.strip() == l.strip() for p, l in zip(predictions, labels)])
    return {
        "wer": wer_score,
        "cer": cer_score,
        "accuracy": accuracy,
    }

def compute_metrics(pred, processor):
    """
    Compute WER, CER, and accuracy for ASR predictions.
    This is the main metrics function used by the Trainer.
    Args:
        pred: Prediction object from Trainer
        processor: The processor/tokenizer for decoding
    Returns:
        dict: WER, CER, accuracy.
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Handle tuple predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    if isinstance(label_ids, tuple):
        label_ids = label_ids[0]

    pred_ids = np.asarray(pred_ids)
    label_ids = np.asarray(label_ids)

    # Get argmax if needed
    if pred_ids.dtype != np.int32 and pred_ids.dtype != np.int64:
        pred_ids = np.argmax(pred_ids, axis=-1)

    # Replace padding tokens
    label_ids = np.where(label_ids == -100, 0, label_ids)

    pred_ids = pred_ids.astype(int).tolist()
    label_ids = label_ids.astype(int).tolist()

    # Handle single examples
    if not isinstance(pred_ids[0], list):
        pred_ids = [pred_ids]
    if not isinstance(label_ids[0], list):
        label_ids = [label_ids]

    # Decode to strings using provided processor
    try:
        if processor is None:
            return {"wer": 1.0, "cer": 1.0, "accuracy": 0.0}
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    except Exception as e:
        return {"wer": 1.0, "cer": 1.0, "accuracy": 0.0}

    if not pred_str or not label_str:
        return {"wer": 1.0, "cer": 1.0, "accuracy": 0.0}

    # Filter out empty strings
    valid_pairs = [(p, l) for p, l in zip(pred_str, label_str) if p.strip() and l.strip()]
    if not valid_pairs:
        return {"wer": 1.0, "cer": 1.0, "accuracy": 0.0}

    pred_str_clean = [p for p, l in valid_pairs]
    label_str_clean = [l for p, l in valid_pairs]

    # Compute metrics
    try:
        wer_score = jiwer.wer(label_str_clean, pred_str_clean)
        cer_score = jiwer.cer(label_str_clean, pred_str_clean)
        accuracy = np.mean([p.strip() == l.strip() for p, l in valid_pairs])
    except Exception as e:
        return {"wer": 1.0, "cer": 1.0, "accuracy": 0.0}

    return {"wer": wer_score, "cer": cer_score, "accuracy": accuracy}

def process_asr_data(train_data, valid_data, processor, config):
    """
    Wrap train/valid data in AutoTrainASRDataset.
    """
    train_dataset = AutoTrainASRDataset(train_data, processor, config)
    valid_dataset = None
    if valid_data is not None:
        valid_dataset = AutoTrainASRDataset(valid_data, processor, config)
    return train_dataset, valid_dataset

def create_asr_model_card(config, trainer):
    """
    Generate a model card for ASR, including validation metrics.
    """
    if getattr(config, "valid_split", None) is not None:
        eval_scores = trainer.evaluate()
        valid_metrics = ASR_EVAL_METRICS
        eval_scores = [f"{k[len('eval_') :]}: {v}" for k, v in eval_scores.items() if k in valid_metrics]
        eval_scores = "\n\n".join(eval_scores)
    else:
        eval_scores = "No validation metrics available"

    if getattr(config, "data_path", None) == f"{config.project_name}/autotrain-data" or os.path.isdir(getattr(config, "data_path", "")):
        dataset_tag = ""
    else:
        dataset_tag = f"\ndatasets:\n- {config.data_path}"

    if os.path.isdir(getattr(config, "model", "")):
        base_model = ""
    else:
        base_model = f"\nbase_model: {config.model}"

    model_card = ASR_MODEL_CARD.format(
        dataset_tag=dataset_tag,
        validation_metrics=eval_scores,
        base_model=base_model,
    )
    return model_card 

class UnderfittingDetectionCallback(TrainerCallback):
    """
    Callback to detect and warn about underfitting during training.
    """
    def __init__(self):
        self.underfitting_warnings = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Check for underfitting after each evaluation."""
        if metrics is None:
            return
        eval_loss = metrics.get("eval_loss", float('inf'))
        eval_wer = metrics.get("eval_wer", float('inf'))
        eval_cer = metrics.get("eval_cer", float('inf'))
        underfitting_indicators = []
        if eval_loss > 3.0:
            underfitting_indicators.append(f"High loss: {eval_loss:.4f}")
        if eval_wer > 0.8:
            underfitting_indicators.append(f"High WER: {eval_wer:.4f}")
        if eval_cer > 0.6:
            underfitting_indicators.append(f"High CER: {eval_cer:.4f}")
        if hasattr(self, 'last_eval_loss') and eval_loss >= self.last_eval_loss:
            underfitting_indicators.append("Loss not decreasing")
        self.last_eval_loss = eval_loss
        if underfitting_indicators:
            warning_msg = f"Underfitting detected at epoch {state.epoch}: {', '.join(underfitting_indicators)}"
            logger.warning(warning_msg)
            self.underfitting_warnings.append(warning_msg)
            logger.info("Suggestions to fix underfitting:")
            logger.info("  - Increase model capacity (use larger model)")
            logger.info("  - Increase learning rate")
            logger.info("  - Train for more epochs")
            logger.info("  - Reduce weight decay")
            logger.info("  - Check data quality and preprocessing")
        else:
            logger.info(f"Good training progress at epoch {state.epoch}: Loss={eval_loss:.4f}, WER={eval_wer:.4f}, CER={eval_cer:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        """Summary at the end of training."""
        if self.underfitting_warnings:
            logger.warning(f"Training completed with {len(self.underfitting_warnings)} underfitting warnings")
            train_dataset_size = getattr(args, 'train_dataset_size', None)
            if train_dataset_size and train_dataset_size < 1000:
                logger.warning("Small dataset detected - consider:")
                logger.warning("  - Using a smaller model to prevent overfitting")
                logger.warning("  - Increasing regularization (weight decay)")
                logger.warning("  - Adding data augmentation")
                logger.warning("  - Using transfer learning from pre-trained model")
            else:
                logger.warning("Large dataset detected - consider:")
                logger.warning("  - Using a larger model for better capacity")
                logger.warning("  - Increasing learning rate")
                logger.warning("  - Training for more epochs")
        else:
            logger.info("Training completed without underfitting issues")