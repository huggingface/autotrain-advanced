APP_AUTOTRAIN_USERNAME = """Please choose the user or organization who is creating the AutoTrain Project.
In case of non-free tier, this user or organization will be billed.
"""

APP_PROJECT_NAME = """A unique name for the AutoTrain Project.
This name will be used to identify the project in the AutoTrain dashboard."""


APP_IMAGE_CLASSIFICATION_DATA_HELP = """The data for the Image Classification task should be in the following format:
- The data should be in a zip file.
- The zip file should contain multiple folders (the classes), each folder should contain images of a single class.
- The name of the folder should be the name of the class.
- The images must be jpeg, jpg or png.
- There should be at least 5 images per class.
- There should not be any other files in the zip file.
- There should not be any other folders inside the zip folder.
"""

APP_LM_TRAINING_TYPE = """There are two types of Language Model Training:
- generic
- chat

In the generic mode, you provide a CSV with a text column which has already been formatted by you for training a language model.
In the chat mode, you provide a CSV with two or three text columns: prompt, context (optional) and response.
Context column can be empty for samples if not needed. You can also have a "prompt start" column. If provided, "prompt start" will be prepended before the prompt column.

Please see [this](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset which has both formats in the same dataset.
"""
