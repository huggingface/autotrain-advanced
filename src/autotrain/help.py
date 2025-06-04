autotrain_user_info = """
<p>Please choose the user or organization who is creating the AutoTrain Project.</p>
<p>In case of non-free tier, this user or organization will be billed.</p>
"""

project_name_info = """A unique name for the AutoTrain Project.
This name will be used to identify the project in the AutoTrain dashboard."""

column_mapping_info = """
<p>Column Mapping is used to map the columns in the dataset to the columns in the AutoTrain Project.</p>
<p>For example, if your dataset has a column named "input" and you want to use it as the input for the model,
you can map it to the "text" column in the AutoTrain Project.</p>
<p>Similarly, if your dataset has a column named "label" and you want to use it as the label for the model,
you can map it to the "target" column in the AutoTrain Project.</p>
<p>Column mapping keys are AutoTrain Project column names and values are your dataset column names.</p>
<p>For tabular datasets, you can map multiple targets to the "label" column. This will enable multi-label task.
The column names must be a comma separated list.</p>
<p>For other tasks, mappings are one-to-one.</p>
<p>Note: column names are case sensitive.</p>
"""

base_model_info = """
<p>Base Model is the model that will be used for fine-tuning.</p>
<p>For example, if you are training a text classification model, you can choose a base model like "bert-base-uncased".</p>
<p>For a list of available models, please see <a href="https://huggingface.co/models" target="_blank">HuggingFace Model Hub</a>.</p>
<p>Note: not all models listed here are going to be compatible with
your data and parameters. You should select a model that is compatible with your task, data and parameters.</p>
Dont see your favorite model? You can also use a custom model by providing the model name in an environment variable: AUTOTRAIN_CUSTOM_MODELS.
For example, go to settings and add a new environment variable with the key AUTOTRAIN_CUSTOM_MODELS and value as the model name (e.g. google/gemma-7b)
"""

hardware_info = """

<p>Hardware is the machine that will be used for training.</p>
<p>Please choose a hardware that is compatible with your task, data and parameters.</p>
"""

task_info = """
<p>Task is the type of model you want to train.</p>
<p>Please choose a task that is compatible with your data and parameters.</p>
<p>For example, if you are training a text classification model, you can choose "Text Classification" task.</p>
"""


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


def get_app_help(element_id):
    if element_id == "autotrain_user_info":
        return autotrain_user_info
    elif element_id == "project_name_info":
        return project_name_info
    elif element_id == "column_mapping_info":
        return column_mapping_info
    elif element_id == "base_model_info":
        return base_model_info
    elif element_id == "hardware_info":
        return hardware_info
    elif element_id == "task_info":
        return task_info
    else:
        return "No help available for this element."
