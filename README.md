# 🤗 AutoNLP

AutoNLP: faster and easier training and deployments of SOTA NLP models

## Installation

You can Install AutoNLP python package via PIP. Please note you will need python >= 3.7 for AutoNLP to work properly.

    pip install autonlp
    
Please make sure that you have git lfs installed. Check out the instructions here: https://github.com/git-lfs/git-lfs/wiki/Installation

## Quick start - in the terminal

Please take a look at [AutoNLP Documentation](https://huggingface.co/docs/autonlp/) for a list of supported tasks and languages.

Note:
AutoNLP is currently in beta release. To participate in the beta, just go to https://huggingface.co/autonlp and apply 🤗

First, create a project:

```bash
autonlp login --api-key YOUR_HUGGING_FACE_API_TOKEN
autonlp create_project --name sentiment_detection --language en --task binary_classification --max_models 5
```

Upload files and start the training. You need a training and a validation split. Only CSV files are supported at the moment.
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
autonlp metrics --project PROJECT_ID
```

## Quick start - Python API

Setting up:
```python
from autonlp import AutoNLP
client = AutoNLP()
client.login(token="YOUR_HUGGING_FACE_API_TOKEN")
```

Creating a project and uploading files to it:
```python
project = client.create_project(name="sentiment_detection", task="binary_classification", language="en", max_models=5)
project.upload(
    filepaths=["/path/to/train.csv"],
    split="train",
    col_mapping={
        "review": "text",
        "sentiment": "target",
    })

# also upload a validation with split="valid"
```

Start the training of your models:
```python
project.train()
```

To monitor the progress of your training:
```python
project.refresh()
print(project)
```

After the training of your models has succeeded, you can retrieve the metrics for each model and test them with the 🤗 Inference API:

```python
client.predict(project="sentiment_detection", model_id=42, input_text="i love autonlp")
```

or use command line:

```bash
autonlp predict --project sentiment_detection --model_id 42 --sentence "i love autonlp"
```

## How much do I have to pay?

It's difficult to provide an exact answer to this question, however, we have an estimator that might help you.
Just enter the number of samples and language and you will get an estimate. Please keep in mind that this is just an estimate and can easily over-estimate or under-estimate (we are actively working on this).

```bash
autonlp estimate --num_train_samples 10000 --project_name sentiment_detection
```
