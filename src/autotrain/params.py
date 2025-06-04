import json
from dataclasses import dataclass
from typing import Optional

from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.extractive_question_answering.params import ExtractiveQuestionAnsweringParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.image_regression.params import ImageRegressionParams
from autotrain.trainers.object_detection.params import ObjectDetectionParams
from autotrain.trainers.sent_transformers.params import SentenceTransformersParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.text_regression.params import TextRegressionParams
from autotrain.trainers.token_classification.params import TokenClassificationParams
from autotrain.trainers.vlm.params import VLMTrainingParams


HIDDEN_PARAMS = [
    "token",
    "project_name",
    "username",
    "task",
    "backend",
    "train_split",
    "valid_split",
    "text_column",
    "rejected_text_column",
    "prompt_text_column",
    "push_to_hub",
    "trainer",
    "model",
    "data_path",
    "image_path",
    "class_image_path",
    "revision",
    "tokenizer",
    "class_prompt",
    "num_class_images",
    "class_labels_conditioning",
    "resume_from_checkpoint",
    "dataloader_num_workers",
    "allow_tf32",
    "prior_generation_precision",
    "local_rank",
    "tokenizer_max_length",
    "rank",
    "xl",
    "checkpoints_total_limit",
    "validation_images",
    "validation_epochs",
    "num_validation_images",
    "validation_prompt",
    "sample_batch_size",
    "log",
    "image_column",
    "target_column",
    "id_column",
    "target_columns",
    "tokens_column",
    "tags_column",
    "objects_column",
    "sentence1_column",
    "sentence2_column",
    "sentence3_column",
    "question_column",
    "answer_column",
]


PARAMS = {}
PARAMS["llm"] = LLMTrainingParams(
    target_modules="all-linear",
    log="tensorboard",
    mixed_precision="fp16",
    quantization="int4",
    peft=True,
    block_size=1024,
    epochs=3,
    padding="right",
    chat_template="none",
    max_completion_length=128,
    distributed_backend="ddp",
).model_dump()

PARAMS["text-classification"] = TextClassificationParams(
    mixed_precision="fp16",
    log="tensorboard",
).model_dump()
PARAMS["st"] = SentenceTransformersParams(
    mixed_precision="fp16",
    log="tensorboard",
).model_dump()
PARAMS["image-classification"] = ImageClassificationParams(
    mixed_precision="fp16",
    log="tensorboard",
).model_dump()
PARAMS["image-object-detection"] = ObjectDetectionParams(
    mixed_precision="fp16",
    log="tensorboard",
).model_dump()
PARAMS["seq2seq"] = Seq2SeqParams(
    mixed_precision="fp16",
    target_modules="all-linear",
    log="tensorboard",
).model_dump()
PARAMS["tabular"] = TabularParams(
    categorical_imputer="most_frequent",
    numerical_imputer="median",
    numeric_scaler="robust",
).model_dump()
PARAMS["token-classification"] = TokenClassificationParams(
    mixed_precision="fp16",
    log="tensorboard",
).model_dump()
PARAMS["text-regression"] = TextRegressionParams(
    mixed_precision="fp16",
    log="tensorboard",
).model_dump()
PARAMS["image-regression"] = ImageRegressionParams(
    mixed_precision="fp16",
    log="tensorboard",
).model_dump()
PARAMS["vlm"] = VLMTrainingParams(
    mixed_precision="fp16",
    target_modules="all-linear",
    log="tensorboard",
    quantization="int4",
    peft=True,
    epochs=3,
).model_dump()
PARAMS["extractive-qa"] = ExtractiveQuestionAnsweringParams(
    mixed_precision="fp16",
    log="tensorboard",
    max_seq_length=512,
    max_doc_stride=128,
).model_dump()


@dataclass
class AppParams:
    """
    AppParams class is responsible for managing and processing parameters for various machine learning tasks.

    Attributes:
        job_params_json (str): JSON string containing job parameters.
        token (str): Authentication token.
        project_name (str): Name of the project.
        username (str): Username of the project owner.
        task (str): Type of task to be performed.
        data_path (str): Path to the dataset.
        base_model (str): Base model to be used.
        column_mapping (dict): Mapping of columns for the dataset.
        train_split (Optional[str]): Name of the training split. Default is None.
        valid_split (Optional[str]): Name of the validation split. Default is None.
        using_hub_dataset (Optional[bool]): Flag indicating if a hub dataset is used. Default is False.
        api (Optional[bool]): Flag indicating if API is used. Default is False.

    Methods:
        __post_init__(): Validates the parameters after initialization.
        munge(): Processes the parameters based on the task type.
        _munge_common_params(): Processes common parameters for all tasks.
        _munge_params_sent_transformers(): Processes parameters for sentence transformers task.
        _munge_params_llm(): Processes parameters for large language model task.
        _munge_params_vlm(): Processes parameters for vision-language model task.
        _munge_params_text_clf(): Processes parameters for text classification task.
        _munge_params_extractive_qa(): Processes parameters for extractive question answering task.
        _munge_params_text_reg(): Processes parameters for text regression task.
        _munge_params_token_clf(): Processes parameters for token classification task.
        _munge_params_seq2seq(): Processes parameters for sequence-to-sequence task.
        _munge_params_img_clf(): Processes parameters for image classification task.
        _munge_params_img_reg(): Processes parameters for image regression task.
        _munge_params_img_obj_det(): Processes parameters for image object detection task.
        _munge_params_tabular(): Processes parameters for tabular data task.
    """

    job_params_json: str
    token: str
    project_name: str
    username: str
    task: str
    data_path: str
    base_model: str
    column_mapping: dict
    train_split: Optional[str] = None
    valid_split: Optional[str] = None
    using_hub_dataset: Optional[bool] = False
    api: Optional[bool] = False

    def __post_init__(self):
        if self.using_hub_dataset and not self.train_split:
            raise ValueError("train_split is required when using a hub dataset")

    def munge(self):
        if self.task == "text-classification":
            return self._munge_params_text_clf()
        elif self.task == "seq2seq":
            return self._munge_params_seq2seq()
        elif self.task == "image-classification":
            return self._munge_params_img_clf()
        elif self.task == "image-object-detection":
            return self._munge_params_img_obj_det()
        elif self.task.startswith("tabular"):
            return self._munge_params_tabular()
        elif self.task.startswith("llm"):
            return self._munge_params_llm()
        elif self.task == "token-classification":
            return self._munge_params_token_clf()
        elif self.task == "text-regression":
            return self._munge_params_text_reg()
        elif self.task.startswith("st:"):
            return self._munge_params_sent_transformers()
        elif self.task == "image-regression":
            return self._munge_params_img_reg()
        elif self.task.startswith("vlm"):
            return self._munge_params_vlm()
        elif self.task == "extractive-qa":
            return self._munge_params_extractive_qa()
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def _munge_common_params(self):
        _params = json.loads(self.job_params_json)
        _params["token"] = self.token
        _params["project_name"] = f"{self.project_name}"
        if "push_to_hub" not in _params:
            _params["push_to_hub"] = True
        _params["data_path"] = self.data_path
        _params["username"] = self.username
        return _params

    def _munge_params_sent_transformers(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        if "log" not in _params:
            _params["log"] = "tensorboard"
        if not self.using_hub_dataset:
            _params["sentence1_column"] = "autotrain_sentence1"
            _params["sentence2_column"] = "autotrain_sentence2"
            _params["sentence3_column"] = "autotrain_sentence3"
            _params["target_column"] = "autotrain_target"
            _params["valid_split"] = "validation"
        else:
            _params["sentence1_column"] = self.column_mapping.get(
                "sentence1" if not self.api else "sentence1_column", "sentence1"
            )
            _params["sentence2_column"] = self.column_mapping.get(
                "sentence2" if not self.api else "sentence2_column", "sentence2"
            )
            _params["sentence3_column"] = self.column_mapping.get(
                "sentence3" if not self.api else "sentence3_column", "sentence3"
            )
            _params["target_column"] = self.column_mapping.get("target" if not self.api else "target_column", "target")
            _params["train_split"] = self.train_split
            _params["valid_split"] = self.valid_split

        trainer = self.task.split(":")[1]
        _params["trainer"] = trainer.lower()
        return SentenceTransformersParams(**_params)

    def _munge_params_llm(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        if not self.using_hub_dataset:
            _params["text_column"] = "autotrain_text"
            _params["prompt_text_column"] = "autotrain_prompt"
            _params["rejected_text_column"] = "autotrain_rejected_text"
        else:
            _params["text_column"] = self.column_mapping.get("text" if not self.api else "text_column", "text")
            _params["prompt_text_column"] = self.column_mapping.get(
                "prompt" if not self.api else "prompt_text_column", "prompt"
            )
            _params["rejected_text_column"] = self.column_mapping.get(
                "rejected_text" if not self.api else "rejected_text_column", "rejected_text"
            )
            _params["train_split"] = self.train_split
        if "log" not in _params:
            _params["log"] = "tensorboard"

        trainer = self.task.split(":")[1]
        if trainer != "generic":
            _params["trainer"] = trainer.lower()

        if "quantization" in _params:
            if _params["quantization"] in ("none", "no"):
                _params["quantization"] = None

        return LLMTrainingParams(**_params)

    def _munge_params_vlm(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        if not self.using_hub_dataset:
            _params["text_column"] = "autotrain_text"
            _params["prompt_text_column"] = "autotrain_prompt"
            _params["image_column"] = "autotrain_image"
            _params["valid_split"] = "validation"
        else:
            _params["text_column"] = self.column_mapping.get("text" if not self.api else "text_column", "text")
            _params["prompt_text_column"] = self.column_mapping.get(
                "prompt" if not self.api else "prompt_text_column", "prompt"
            )
            _params["image_column"] = self.column_mapping.get(
                "image" if not self.api else "rejected_text_column", "image"
            )
            _params["train_split"] = self.train_split
            _params["valid_split"] = self.valid_split
        if "log" not in _params:
            _params["log"] = "tensorboard"

        trainer = self.task.split(":")[1]
        _params["trainer"] = trainer.lower()

        if "quantization" in _params:
            if _params["quantization"] in ("none", "no"):
                _params["quantization"] = None

        return VLMTrainingParams(**_params)

    def _munge_params_text_clf(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        if "log" not in _params:
            _params["log"] = "tensorboard"
        if not self.using_hub_dataset:
            _params["text_column"] = "autotrain_text"
            _params["target_column"] = "autotrain_label"
            _params["valid_split"] = "validation"
        else:
            _params["text_column"] = self.column_mapping.get("text" if not self.api else "text_column", "text")
            _params["target_column"] = self.column_mapping.get("label" if not self.api else "target_column", "label")
            _params["train_split"] = self.train_split
            _params["valid_split"] = self.valid_split
        return TextClassificationParams(**_params)

    def _munge_params_extractive_qa(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        if "log" not in _params:
            _params["log"] = "tensorboard"
        if not self.using_hub_dataset:
            _params["text_column"] = "autotrain_text"
            _params["question_column"] = "autotrain_question"
            _params["answer_column"] = "autotrain_answer"
            _params["valid_split"] = "validation"
        else:
            _params["text_column"] = self.column_mapping.get("text" if not self.api else "text_column", "text")
            _params["question_column"] = self.column_mapping.get(
                "question" if not self.api else "question_column", "question"
            )
            _params["answer_column"] = self.column_mapping.get("answer" if not self.api else "answer_column", "answer")
            _params["train_split"] = self.train_split
            _params["valid_split"] = self.valid_split
        return ExtractiveQuestionAnsweringParams(**_params)

    def _munge_params_text_reg(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        if "log" not in _params:
            _params["log"] = "tensorboard"
        if not self.using_hub_dataset:
            _params["text_column"] = "autotrain_text"
            _params["target_column"] = "autotrain_label"
            _params["valid_split"] = "validation"
        else:
            _params["text_column"] = self.column_mapping.get("text" if not self.api else "text_column", "text")
            _params["target_column"] = self.column_mapping.get("label" if not self.api else "target_column", "label")
            _params["train_split"] = self.train_split
            _params["valid_split"] = self.valid_split
        return TextRegressionParams(**_params)

    def _munge_params_token_clf(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        if "log" not in _params:
            _params["log"] = "tensorboard"
        if not self.using_hub_dataset:
            _params["tokens_column"] = "autotrain_text"
            _params["tags_column"] = "autotrain_label"
            _params["valid_split"] = "validation"
        else:
            _params["tokens_column"] = self.column_mapping.get("tokens" if not self.api else "tokens_column", "tokens")
            _params["tags_column"] = self.column_mapping.get("tags" if not self.api else "tags_column", "tags")
            _params["train_split"] = self.train_split
            _params["valid_split"] = self.valid_split

        return TokenClassificationParams(**_params)

    def _munge_params_seq2seq(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        if "log" not in _params:
            _params["log"] = "tensorboard"
        if not self.using_hub_dataset:
            _params["text_column"] = "autotrain_text"
            _params["target_column"] = "autotrain_label"
            _params["valid_split"] = "validation"
        else:
            _params["text_column"] = self.column_mapping.get("text" if not self.api else "text_column", "text")
            _params["target_column"] = self.column_mapping.get("label" if not self.api else "target_column", "label")
            _params["train_split"] = self.train_split
            _params["valid_split"] = self.valid_split

        return Seq2SeqParams(**_params)

    def _munge_params_img_clf(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        if "log" not in _params:
            _params["log"] = "tensorboard"
        if not self.using_hub_dataset:
            _params["image_column"] = "autotrain_image"
            _params["target_column"] = "autotrain_label"
            _params["valid_split"] = "validation"
        else:
            _params["image_column"] = self.column_mapping.get("image" if not self.api else "image_column", "image")
            _params["target_column"] = self.column_mapping.get("label" if not self.api else "target_column", "label")
            _params["train_split"] = self.train_split
            _params["valid_split"] = self.valid_split

        return ImageClassificationParams(**_params)

    def _munge_params_img_reg(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        if "log" not in _params:
            _params["log"] = "tensorboard"
        if not self.using_hub_dataset:
            _params["image_column"] = "autotrain_image"
            _params["target_column"] = "autotrain_label"
            _params["valid_split"] = "validation"
        else:
            _params["image_column"] = self.column_mapping.get("image" if not self.api else "image_column", "image")
            _params["target_column"] = self.column_mapping.get("target" if not self.api else "target_column", "target")
            _params["train_split"] = self.train_split
            _params["valid_split"] = self.valid_split

        return ImageRegressionParams(**_params)

    def _munge_params_img_obj_det(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        if "log" not in _params:
            _params["log"] = "tensorboard"
        if not self.using_hub_dataset:
            _params["image_column"] = "autotrain_image"
            _params["objects_column"] = "autotrain_objects"
            _params["valid_split"] = "validation"
        else:
            _params["image_column"] = self.column_mapping.get("image" if not self.api else "image_column", "image")
            _params["objects_column"] = self.column_mapping.get(
                "objects" if not self.api else "objects_column", "objects"
            )
            _params["train_split"] = self.train_split
            _params["valid_split"] = self.valid_split

        return ObjectDetectionParams(**_params)

    def _munge_params_tabular(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        if not self.using_hub_dataset:
            _params["id_column"] = "autotrain_id"
            _params["valid_split"] = "validation"
            if len(self.column_mapping["label"]) == 1:
                _params["target_columns"] = ["autotrain_label"]
            else:
                _params["target_columns"] = [
                    "autotrain_label_" + str(i) for i in range(len(self.column_mapping["label"]))
                ]
        else:
            _params["id_column"] = self.column_mapping.get("id" if not self.api else "id_column", "id")
            _params["train_split"] = self.train_split
            _params["valid_split"] = self.valid_split
            _params["target_columns"] = self.column_mapping.get("label" if not self.api else "target_columns", "label")

        if len(_params["categorical_imputer"].strip()) == 0 or _params["categorical_imputer"].lower() == "none":
            _params["categorical_imputer"] = None
        if len(_params["numerical_imputer"].strip()) == 0 or _params["numerical_imputer"].lower() == "none":
            _params["numerical_imputer"] = None
        if len(_params["numeric_scaler"].strip()) == 0 or _params["numeric_scaler"].lower() == "none":
            _params["numeric_scaler"] = None

        if "classification" in self.task:
            _params["task"] = "classification"
        else:
            _params["task"] = "regression"

        return TabularParams(**_params)


def get_task_params(task, param_type):
    """
    Retrieve task-specific parameters while filtering out hidden parameters based on the task and parameter type.

    Args:
        task (str): The task identifier, which can include prefixes like "llm", "st:", "vlm:", etc.
        param_type (str): The type of parameters to retrieve, typically "basic" or other types.

    Returns:
        dict: A dictionary of task-specific parameters with hidden parameters filtered out.

    Notes:
        - The function handles various task prefixes and adjusts the task and trainer variables accordingly.
        - Hidden parameters are filtered out based on the task and parameter type.
        - Additional hidden parameters are defined for specific tasks and trainers.
    """
    if task.startswith("llm"):
        trainer = task.split(":")[1].lower()
        task = task.split(":")[0].lower()

    if task.startswith("st:"):
        trainer = task.split(":")[1].lower()
        task = task.split(":")[0].lower()

    if task.startswith("vlm:"):
        trainer = task.split(":")[1].lower()
        task = task.split(":")[0].lower()

    if task.startswith("tabular"):
        task = "tabular"

    if task not in PARAMS:
        return {}

    task_params = PARAMS[task]
    task_params = {k: v for k, v in task_params.items() if k not in HIDDEN_PARAMS}
    if task == "llm":
        more_hidden_params = []
        if trainer == "sft":
            more_hidden_params = [
                "model_ref",
                "dpo_beta",
                "add_eos_token",
                "max_prompt_length",
                "max_completion_length",
            ]
        elif trainer == "reward":
            more_hidden_params = [
                "model_ref",
                "dpo_beta",
                "add_eos_token",
                "max_prompt_length",
                "max_completion_length",
                "unsloth",
            ]
        elif trainer == "orpo":
            more_hidden_params = [
                "model_ref",
                "dpo_beta",
                "add_eos_token",
                "unsloth",
            ]
        elif trainer == "generic":
            more_hidden_params = [
                "model_ref",
                "dpo_beta",
                "max_prompt_length",
                "max_completion_length",
            ]
        elif trainer == "dpo":
            more_hidden_params = [
                "add_eos_token",
                "unsloth",
            ]
        if param_type == "basic":
            more_hidden_params.extend(
                [
                    "padding",
                    "use_flash_attention_2",
                    "disable_gradient_checkpointing",
                    "logging_steps",
                    "eval_strategy",
                    "save_total_limit",
                    "auto_find_batch_size",
                    "warmup_ratio",
                    "weight_decay",
                    "max_grad_norm",
                    "seed",
                    "quantization",
                    "merge_adapter",
                    "lora_r",
                    "lora_alpha",
                    "lora_dropout",
                    "max_completion_length",
                ]
            )
        task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}
    if task == "text-classification" and param_type == "basic":
        more_hidden_params = [
            "warmup_ratio",
            "weight_decay",
            "max_grad_norm",
            "seed",
            "logging_steps",
            "auto_find_batch_size",
            "save_total_limit",
            "eval_strategy",
            "early_stopping_patience",
            "early_stopping_threshold",
        ]
        task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}
    if task == "extractive-qa" and param_type == "basic":
        more_hidden_params = [
            "warmup_ratio",
            "weight_decay",
            "max_grad_norm",
            "seed",
            "logging_steps",
            "auto_find_batch_size",
            "save_total_limit",
            "eval_strategy",
            "early_stopping_patience",
            "early_stopping_threshold",
        ]
        task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}
    if task == "st" and param_type == "basic":
        more_hidden_params = [
            "warmup_ratio",
            "weight_decay",
            "max_grad_norm",
            "seed",
            "logging_steps",
            "auto_find_batch_size",
            "save_total_limit",
            "eval_strategy",
            "early_stopping_patience",
            "early_stopping_threshold",
        ]
        task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}
    if task == "vlm" and param_type == "basic":
        more_hidden_params = [
            "warmup_ratio",
            "weight_decay",
            "max_grad_norm",
            "seed",
            "logging_steps",
            "auto_find_batch_size",
            "save_total_limit",
            "eval_strategy",
            "early_stopping_patience",
            "early_stopping_threshold",
            "quantization",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
        ]
        task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}
    if task == "text-regression" and param_type == "basic":
        more_hidden_params = [
            "warmup_ratio",
            "weight_decay",
            "max_grad_norm",
            "seed",
            "logging_steps",
            "auto_find_batch_size",
            "save_total_limit",
            "eval_strategy",
            "early_stopping_patience",
            "early_stopping_threshold",
        ]
        task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}
    if task == "image-classification" and param_type == "basic":
        more_hidden_params = [
            "warmup_ratio",
            "weight_decay",
            "max_grad_norm",
            "seed",
            "logging_steps",
            "auto_find_batch_size",
            "save_total_limit",
            "eval_strategy",
            "early_stopping_patience",
            "early_stopping_threshold",
        ]
        task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}
    if task == "image-regression" and param_type == "basic":
        more_hidden_params = [
            "warmup_ratio",
            "weight_decay",
            "max_grad_norm",
            "seed",
            "logging_steps",
            "auto_find_batch_size",
            "save_total_limit",
            "eval_strategy",
            "early_stopping_patience",
            "early_stopping_threshold",
        ]
        task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}
    if task == "image-object-detection" and param_type == "basic":
        more_hidden_params = [
            "warmup_ratio",
            "weight_decay",
            "max_grad_norm",
            "seed",
            "logging_steps",
            "auto_find_batch_size",
            "save_total_limit",
            "eval_strategy",
            "early_stopping_patience",
            "early_stopping_threshold",
        ]
        task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}
    if task == "seq2seq" and param_type == "basic":
        more_hidden_params = [
            "warmup_ratio",
            "weight_decay",
            "max_grad_norm",
            "seed",
            "logging_steps",
            "auto_find_batch_size",
            "save_total_limit",
            "eval_strategy",
            "quantization",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "early_stopping_patience",
            "early_stopping_threshold",
        ]
        task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}
    if task == "token-classification" and param_type == "basic":
        more_hidden_params = [
            "warmup_ratio",
            "weight_decay",
            "max_grad_norm",
            "seed",
            "logging_steps",
            "auto_find_batch_size",
            "save_total_limit",
            "eval_strategy",
            "early_stopping_patience",
            "early_stopping_threshold",
        ]
        task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}

    return task_params
