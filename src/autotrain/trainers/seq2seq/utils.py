import evaluate
import nltk
import numpy as np


ROUGE_METRIC = evaluate.load("rouge")

MODEL_CARD = """
---
tags:
- autotrain
- text2text-generation
widget:
- text: "I love AutoTrain"
datasets:
- {dataset}
---

# Model Trained Using AutoTrain

- Problem type: Seq2Seq

## Validation Metrics
{validation_metrics}
"""


def _seq2seq_metrics(pred, tokenizer):
    predictions, labels = pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = ROUGE_METRIC.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def create_model_card(config, trainer):
    if config.valid_split is not None:
        eval_scores = trainer.evaluate()
        eval_scores = [f"{k[len('eval_'):]}: {v}" for k, v in eval_scores.items()]
        eval_scores = "\n\n".join(eval_scores)

    else:
        eval_scores = "No validation metrics available"

    model_card = MODEL_CARD.format(
        dataset=config.data_path,
        validation_metrics=eval_scores,
    )
    return model_card
