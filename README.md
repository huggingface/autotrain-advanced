# AutoNLP

AutoNLP: faster and easier training and deployments of SOTA NLP models

## Installation

Install AutoNLP python package

    pip install .

## Quick start - in the terminal

Supported languages:

- English: en
- French: fr
- German: de
- Spanish: es
- Finnish: fi

Supported tasks:

- binary_classification
- multi_class_classification
- entity_extraction

First, create a project:

```bash
autonlp login --api-key YOUR_HF_API_TOKEN
autonlp create_project --name sentiment_detection --language en --task binary_classification
```

Upload files and start the training. Only CSV files are supported at the moment.
```bash
# Train split
autonlp upload --project sentiment_detection --split train \
               --col_mapping review:text,sentiment:target \
               --files ~/datasets/train.csv
# Validation split
autonlp upload --project sentiment_detection --split valid \
               --col_mapping review:text,sentiment:target \
               --files ~/datasets/valid.csv
```

Once the files are uploaded, you can start training the model:
```bash
autonlp train --project sentiment_detection
```

Monitor the progress of your project.
```bash
# Project progress
autonlp project_info --name sentiment_detection
# Model metrics
autonlp model_info --id MODEL_ID
```

## Quick start - Python API

Setting up:
```python
from autonlp import AutoNLP
client = AutoNLP()
client.login(token="YOUR_HF_API_TOKEN")
```

Creating a project and uploading files to it:
```python
project = client.create_project(name="sentiment_detection", task="binary_classification", language="en")
project.upload(
    filepaths=["/path/to/train.csv"],
    split="train",
    col_mapping={
        "Title": "text",
        "Conference": "target",
    })
```

Start the training and monitor the progress:
```python
project.train()
project.refresh()
print(project)
```

After the training of your models has succeeded, you can retrieve its metrics and test it with the 🤗 Inference API:

```python
client.get_model_info(model_id=42)
client.predict(model_id=42, input_text="Measuring and Improving Consistency in Pretrained Language Models")
```
