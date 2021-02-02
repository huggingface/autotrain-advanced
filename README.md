# AutoNLP

AutoNLP: faster and easier training and deployments of NLP models

## Installation

Install AutoNLP python package

    pip install .

## Quick start - in the terminal

First, create a project. Only `fr`, `en` languages and `binary_classification`, `multi_class_classification` are supported at the moment.
```bash
autonlp login --api-key YOUR_HF_API_TOKEN
autonlp create_project --name test_project --language en --task multi_class_classification
```

Upload files and start the training. Only CSV files are supported at the moment.
```bash
# Train split
autonlp upload --project test_project --split train\
               --col_mapping Title:text,Conference:target\
               --files ~/datasets/title_conf_train.csv
# Validation split
autonlp upload --project test_project --split valid\
               --col_mapping Title:text,Conference:target\
               --files ~/datasets/title_conf_valid.csv
autonlp train --project test_project
```

Monitor the progress of your project.
```bash
# Project progress
autonlp project_info --name test_project
# Model metrics
autonlp model_info --id MODEL_ID
```

## Quick start - Python API

Setting up
```python
from autonlp import AutoNLP
client = AutoNLP()
client.login(token="YOUR_HF_API_TOKEN")
```

Creating a project and uploading files to it:
```python
project = client.create_project(name="test_project", task="multi_class_classification", language="fr")
project.upload(
    filepaths=["itle_conf_train.csv"],
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

After the training of your models has succeeded, you can test it with the 🤗 Inference API:
```python
client.predict(model_id=42, input_text="Measuring and Improving Consistency in Pretrained Language Models")
```
