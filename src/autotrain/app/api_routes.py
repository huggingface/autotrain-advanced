import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, get_type_hints

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from huggingface_hub import HfApi, constants
from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status
from pydantic import BaseModel, create_model, model_validator

from autotrain import __version__, logger
from autotrain.app.params import HIDDEN_PARAMS, PARAMS, AppParams
from autotrain.app.utils import token_verification
from autotrain.project import AutoTrainProject
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


FIELDS_TO_EXCLUDE = HIDDEN_PARAMS + ["push_to_hub"]


def create_api_base_model(base_class, class_name):
    """
    Creates a new Pydantic model based on a given base class and class name,
    excluding specified fields.

    Args:
        base_class (Type): The base Pydantic model class to extend.
        class_name (str): The name of the new model class to create.

    Returns:
        Type: A new Pydantic model class with the specified modifications.

    Notes:
        - The function uses type hints from the base class to define the new model's fields.
        - Certain fields are excluded from the new model based on the class name.
        - The function supports different sets of hidden parameters for different class names.
        - The new model's configuration is set to have no protected namespaces.
    """
    annotations = get_type_hints(base_class)
    if class_name in ("LLMSFTTrainingParamsAPI", "LLMRewardTrainingParamsAPI"):
        more_hidden_params = [
            "model_ref",
            "dpo_beta",
            "add_eos_token",
            "max_prompt_length",
            "max_completion_length",
        ]
    elif class_name == "LLMORPOTrainingParamsAPI":
        more_hidden_params = [
            "model_ref",
            "dpo_beta",
            "add_eos_token",
        ]
    elif class_name == "LLMDPOTrainingParamsAPI":
        more_hidden_params = [
            "add_eos_token",
        ]
    elif class_name == "LLMGenericTrainingParamsAPI":
        more_hidden_params = [
            "model_ref",
            "dpo_beta",
            "max_prompt_length",
            "max_completion_length",
        ]
    else:
        more_hidden_params = []
    _excluded = FIELDS_TO_EXCLUDE + more_hidden_params
    new_fields: Dict[str, Tuple[Any, Any]] = {}
    for name, field in base_class.__fields__.items():
        if name not in _excluded:
            field_type = annotations[name]
            if field.default is not None:
                field_default = field.default
            elif field.default_factory is not None:
                field_default = field.default_factory
            else:
                field_default = None
            new_fields[name] = (field_type, field_default)
    return create_model(
        class_name,
        **{key: (value[0], value[1]) for key, value in new_fields.items()},
        __config__=type("Config", (), {"protected_namespaces": ()}),
    )


LLMSFTTrainingParamsAPI = create_api_base_model(LLMTrainingParams, "LLMSFTTrainingParamsAPI")
LLMDPOTrainingParamsAPI = create_api_base_model(LLMTrainingParams, "LLMDPOTrainingParamsAPI")
LLMORPOTrainingParamsAPI = create_api_base_model(LLMTrainingParams, "LLMORPOTrainingParamsAPI")
LLMGenericTrainingParamsAPI = create_api_base_model(LLMTrainingParams, "LLMGenericTrainingParamsAPI")
LLMRewardTrainingParamsAPI = create_api_base_model(LLMTrainingParams, "LLMRewardTrainingParamsAPI")
ImageClassificationParamsAPI = create_api_base_model(ImageClassificationParams, "ImageClassificationParamsAPI")
Seq2SeqParamsAPI = create_api_base_model(Seq2SeqParams, "Seq2SeqParamsAPI")
TabularClassificationParamsAPI = create_api_base_model(TabularParams, "TabularClassificationParamsAPI")
TabularRegressionParamsAPI = create_api_base_model(TabularParams, "TabularRegressionParamsAPI")
TextClassificationParamsAPI = create_api_base_model(TextClassificationParams, "TextClassificationParamsAPI")
TextRegressionParamsAPI = create_api_base_model(TextRegressionParams, "TextRegressionParamsAPI")
TokenClassificationParamsAPI = create_api_base_model(TokenClassificationParams, "TokenClassificationParamsAPI")
SentenceTransformersParamsAPI = create_api_base_model(SentenceTransformersParams, "SentenceTransformersParamsAPI")
ImageRegressionParamsAPI = create_api_base_model(ImageRegressionParams, "ImageRegressionParamsAPI")
VLMTrainingParamsAPI = create_api_base_model(VLMTrainingParams, "VLMTrainingParamsAPI")
ExtractiveQuestionAnsweringParamsAPI = create_api_base_model(
    ExtractiveQuestionAnsweringParams, "ExtractiveQuestionAnsweringParamsAPI"
)
ObjectDetectionParamsAPI = create_api_base_model(ObjectDetectionParams, "ObjectDetectionParamsAPI")


class LLMSFTColumnMapping(BaseModel):
    text_column: str


class LLMDPOColumnMapping(BaseModel):
    text_column: str
    rejected_text_column: str
    prompt_text_column: str


class LLMORPOColumnMapping(BaseModel):
    text_column: str
    rejected_text_column: str
    prompt_text_column: str


class LLMGenericColumnMapping(BaseModel):
    text_column: str


class LLMRewardColumnMapping(BaseModel):
    text_column: str
    rejected_text_column: str


class ImageClassificationColumnMapping(BaseModel):
    image_column: str
    target_column: str


class ImageRegressionColumnMapping(BaseModel):
    image_column: str
    target_column: str


class Seq2SeqColumnMapping(BaseModel):
    text_column: str
    target_column: str


class TabularClassificationColumnMapping(BaseModel):
    id_column: str
    target_columns: List[str]


class TabularRegressionColumnMapping(BaseModel):
    id_column: str
    target_columns: List[str]


class TextClassificationColumnMapping(BaseModel):
    text_column: str
    target_column: str


class TextRegressionColumnMapping(BaseModel):
    text_column: str
    target_column: str


class TokenClassificationColumnMapping(BaseModel):
    tokens_column: str
    tags_column: str


class STPairColumnMapping(BaseModel):
    sentence1_column: str
    sentence2_column: str


class STPairClassColumnMapping(BaseModel):
    sentence1_column: str
    sentence2_column: str
    target_column: str


class STPairScoreColumnMapping(BaseModel):
    sentence1_column: str
    sentence2_column: str
    target_column: str


class STTripletColumnMapping(BaseModel):
    sentence1_column: str
    sentence2_column: str
    sentence3_column: str


class STQAColumnMapping(BaseModel):
    sentence1_column: str
    sentence2_column: str


class VLMColumnMapping(BaseModel):
    image_column: str
    text_column: str
    prompt_text_column: str


class ExtractiveQuestionAnsweringColumnMapping(BaseModel):
    text_column: str
    question_column: str
    answer_column: str


class ObjectDetectionColumnMapping(BaseModel):
    image_column: str
    objects_column: str


class APICreateProjectModel(BaseModel):
    """
    APICreateProjectModel is a Pydantic model that defines the schema for creating a project.

    Attributes:
        project_name (str): The name of the project.
        task (Literal): The type of task for the project. Supported tasks include various LLM tasks,
            image classification, seq2seq, token classification, text classification,
            text regression, tabular classification, tabular regression, image regression, VLM tasks,
            and extractive question answering.
        base_model (str): The base model to be used for the project.
        hardware (Literal): The type of hardware to be used for the project. Supported hardware options
            include various configurations of spaces and local.
        params (Union): The training parameters for the project. The type of parameters depends on the
            task selected.
        username (str): The username of the person creating the project.
        column_mapping (Optional[Union]): The column mapping for the project. The type of column mapping
            depends on the task selected.
        hub_dataset (str): The dataset to be used for the project.
        train_split (str): The training split of the dataset.
        valid_split (Optional[str]): The validation split of the dataset.

    Methods:
        validate_column_mapping(cls, values): Validates the column mapping based on the task selected.
        validate_params(cls, values): Validates the training parameters based on the task selected.
    """

    project_name: str
    task: Literal[
        "llm:sft",
        "llm:dpo",
        "llm:orpo",
        "llm:generic",
        "llm:reward",
        "st:pair",
        "st:pair_class",
        "st:pair_score",
        "st:triplet",
        "st:qa",
        "image-classification",
        "seq2seq",
        "token-classification",
        "text-classification",
        "text-regression",
        "tabular-classification",
        "tabular-regression",
        "image-regression",
        "vlm:captioning",
        "vlm:vqa",
        "extractive-question-answering",
        "image-object-detection",
    ]
    base_model: str
    hardware: Literal[
        "spaces-a10g-large",
        "spaces-a10g-small",
        "spaces-a100-large",
        "spaces-t4-medium",
        "spaces-t4-small",
        "spaces-cpu-upgrade",
        "spaces-cpu-basic",
        "spaces-l4x1",
        "spaces-l4x4",
        "spaces-l40sx1",
        "spaces-l40sx4",
        "spaces-l40sx8",
        "spaces-a10g-largex2",
        "spaces-a10g-largex4",
        # "local",
    ]
    params: Union[
        LLMSFTTrainingParamsAPI,
        LLMDPOTrainingParamsAPI,
        LLMORPOTrainingParamsAPI,
        LLMGenericTrainingParamsAPI,
        LLMRewardTrainingParamsAPI,
        SentenceTransformersParamsAPI,
        ImageClassificationParamsAPI,
        Seq2SeqParamsAPI,
        TabularClassificationParamsAPI,
        TabularRegressionParamsAPI,
        TextClassificationParamsAPI,
        TextRegressionParamsAPI,
        TokenClassificationParamsAPI,
        ImageRegressionParamsAPI,
        VLMTrainingParamsAPI,
        ExtractiveQuestionAnsweringParamsAPI,
        ObjectDetectionParamsAPI,
    ]
    username: str
    column_mapping: Optional[
        Union[
            LLMSFTColumnMapping,
            LLMDPOColumnMapping,
            LLMORPOColumnMapping,
            LLMGenericColumnMapping,
            LLMRewardColumnMapping,
            ImageClassificationColumnMapping,
            Seq2SeqColumnMapping,
            TabularClassificationColumnMapping,
            TabularRegressionColumnMapping,
            TextClassificationColumnMapping,
            TextRegressionColumnMapping,
            TokenClassificationColumnMapping,
            STPairColumnMapping,
            STPairClassColumnMapping,
            STPairScoreColumnMapping,
            STTripletColumnMapping,
            STQAColumnMapping,
            ImageRegressionColumnMapping,
            VLMColumnMapping,
            ExtractiveQuestionAnsweringColumnMapping,
            ObjectDetectionColumnMapping,
        ]
    ] = None
    hub_dataset: str
    train_split: str
    valid_split: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_column_mapping(cls, values):
        if values.get("task") == "llm:sft":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for llm:sft")
            if not values.get("column_mapping").get("text_column"):
                raise ValueError("text_column is required for llm:sft")
            values["column_mapping"] = LLMSFTColumnMapping(**values["column_mapping"])
        elif values.get("task") == "llm:dpo":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for llm:dpo")
            if not values.get("column_mapping").get("text_column"):
                raise ValueError("text_column is required for llm:dpo")
            if not values.get("column_mapping").get("rejected_text_column"):
                raise ValueError("rejected_text_column is required for llm:dpo")
            if not values.get("column_mapping").get("prompt_text_column"):
                raise ValueError("prompt_text_column is required for llm:dpo")
            values["column_mapping"] = LLMDPOColumnMapping(**values["column_mapping"])
        elif values.get("task") == "llm:orpo":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for llm:orpo")
            if not values.get("column_mapping").get("text_column"):
                raise ValueError("text_column is required for llm:orpo")
            if not values.get("column_mapping").get("rejected_text_column"):
                raise ValueError("rejected_text_column is required for llm:orpo")
            if not values.get("column_mapping").get("prompt_text_column"):
                raise ValueError("prompt_text_column is required for llm:orpo")
            values["column_mapping"] = LLMORPOColumnMapping(**values["column_mapping"])
        elif values.get("task") == "llm:generic":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for llm:generic")
            if not values.get("column_mapping").get("text_column"):
                raise ValueError("text_column is required for llm:generic")
            values["column_mapping"] = LLMGenericColumnMapping(**values["column_mapping"])
        elif values.get("task") == "llm:reward":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for llm:reward")
            if not values.get("column_mapping").get("text_column"):
                raise ValueError("text_column is required for llm:reward")
            if not values.get("column_mapping").get("rejected_text_column"):
                raise ValueError("rejected_text_column is required for llm:reward")
            values["column_mapping"] = LLMRewardColumnMapping(**values["column_mapping"])
        elif values.get("task") == "seq2seq":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for seq2seq")
            if not values.get("column_mapping").get("text_column"):
                raise ValueError("text_column is required for seq2seq")
            if not values.get("column_mapping").get("target_column"):
                raise ValueError("target_column is required for seq2seq")
            values["column_mapping"] = Seq2SeqColumnMapping(**values["column_mapping"])
        elif values.get("task") == "image-classification":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for image-classification")
            if not values.get("column_mapping").get("image_column"):
                raise ValueError("image_column is required for image-classification")
            if not values.get("column_mapping").get("target_column"):
                raise ValueError("target_column is required for image-classification")
            values["column_mapping"] = ImageClassificationColumnMapping(**values["column_mapping"])
        elif values.get("task") == "tabular-classification":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for tabular-classification")
            if not values.get("column_mapping").get("id_column"):
                raise ValueError("id_column is required for tabular-classification")
            if not values.get("column_mapping").get("target_columns"):
                raise ValueError("target_columns is required for tabular-classification")
            values["column_mapping"] = TabularClassificationColumnMapping(**values["column_mapping"])
        elif values.get("task") == "tabular-regression":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for tabular-regression")
            if not values.get("column_mapping").get("id_column"):
                raise ValueError("id_column is required for tabular-regression")
            if not values.get("column_mapping").get("target_columns"):
                raise ValueError("target_columns is required for tabular-regression")
            values["column_mapping"] = TabularRegressionColumnMapping(**values["column_mapping"])
        elif values.get("task") == "text-classification":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for text-classification")
            if not values.get("column_mapping").get("text_column"):
                raise ValueError("text_column is required for text-classification")
            if not values.get("column_mapping").get("target_column"):
                raise ValueError("target_column is required for text-classification")
            values["column_mapping"] = TextClassificationColumnMapping(**values["column_mapping"])
        elif values.get("task") == "text-regression":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for text-regression")
            if not values.get("column_mapping").get("text_column"):
                raise ValueError("text_column is required for text-regression")
            if not values.get("column_mapping").get("target_column"):
                raise ValueError("target_column is required for text-regression")
            values["column_mapping"] = TextRegressionColumnMapping(**values["column_mapping"])
        elif values.get("task") == "token-classification":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for token-classification")
            if not values.get("column_mapping").get("tokens_column"):
                raise ValueError("tokens_column is required for token-classification")
            if not values.get("column_mapping").get("tags_column"):
                raise ValueError("tags_column is required for token-classification")
            values["column_mapping"] = TokenClassificationColumnMapping(**values["column_mapping"])
        elif values.get("task") == "st:pair":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for st:pair")
            if not values.get("column_mapping").get("sentence1_column"):
                raise ValueError("sentence1_column is required for st:pair")
            if not values.get("column_mapping").get("sentence2_column"):
                raise ValueError("sentence2_column is required for st:pair")
            values["column_mapping"] = STPairColumnMapping(**values["column_mapping"])
        elif values.get("task") == "st:pair_class":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for st:pair_class")
            if not values.get("column_mapping").get("sentence1_column"):
                raise ValueError("sentence1_column is required for st:pair_class")
            if not values.get("column_mapping").get("sentence2_column"):
                raise ValueError("sentence2_column is required for st:pair_class")
            if not values.get("column_mapping").get("target_column"):
                raise ValueError("target_column is required for st:pair_class")
            values["column_mapping"] = STPairClassColumnMapping(**values["column_mapping"])
        elif values.get("task") == "st:pair_score":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for st:pair_score")
            if not values.get("column_mapping").get("sentence1_column"):
                raise ValueError("sentence1_column is required for st:pair_score")
            if not values.get("column_mapping").get("sentence2_column"):
                raise ValueError("sentence2_column is required for st:pair_score")
            if not values.get("column_mapping").get("target_column"):
                raise ValueError("target_column is required for st:pair_score")
            values["column_mapping"] = STPairScoreColumnMapping(**values["column_mapping"])
        elif values.get("task") == "st:triplet":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for st:triplet")
            if not values.get("column_mapping").get("sentence1_column"):
                raise ValueError("sentence1_column is required for st:triplet")
            if not values.get("column_mapping").get("sentence2_column"):
                raise ValueError("sentence2_column is required for st:triplet")
            if not values.get("column_mapping").get("sentence3_column"):
                raise ValueError("sentence3_column is required for st:triplet")
            values["column_mapping"] = STTripletColumnMapping(**values["column_mapping"])
        elif values.get("task") == "st:qa":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for st:qa")
            if not values.get("column_mapping").get("sentence1_column"):
                raise ValueError("sentence1_column is required for st:qa")
            if not values.get("column_mapping").get("sentence2_column"):
                raise ValueError("sentence2_column is required for st:qa")
            values["column_mapping"] = STQAColumnMapping(**values["column_mapping"])
        elif values.get("task") == "image-regression":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for image-regression")
            if not values.get("column_mapping").get("image_column"):
                raise ValueError("image_column is required for image-regression")
            if not values.get("column_mapping").get("target_column"):
                raise ValueError("target_column is required for image-regression")
            values["column_mapping"] = ImageRegressionColumnMapping(**values["column_mapping"])
        elif values.get("task") == "vlm:captioning":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for vlm:captioning")
            if not values.get("column_mapping").get("image_column"):
                raise ValueError("image_column is required for vlm:captioning")
            if not values.get("column_mapping").get("text_column"):
                raise ValueError("text_column is required for vlm:captioning")
            if not values.get("column_mapping").get("prompt_text_column"):
                raise ValueError("prompt_text_column is required for vlm:captioning")
            values["column_mapping"] = VLMColumnMapping(**values["column_mapping"])
        elif values.get("task") == "vlm:vqa":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for vlm:vqa")
            if not values.get("column_mapping").get("image_column"):
                raise ValueError("image_column is required for vlm:vqa")
            if not values.get("column_mapping").get("text_column"):
                raise ValueError("text_column is required for vlm:vqa")
            if not values.get("column_mapping").get("prompt_text_column"):
                raise ValueError("prompt_text_column is required for vlm:vqa")
            values["column_mapping"] = VLMColumnMapping(**values["column_mapping"])
        elif values.get("task") == "extractive-question-answering":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for extractive-question-answering")
            if not values.get("column_mapping").get("text_column"):
                raise ValueError("text_column is required for extractive-question-answering")
            if not values.get("column_mapping").get("question_column"):
                raise ValueError("question_column is required for extractive-question-answering")
            if not values.get("column_mapping").get("answer_column"):
                raise ValueError("answer_column is required for extractive-question-answering")
            values["column_mapping"] = ExtractiveQuestionAnsweringColumnMapping(**values["column_mapping"])
        elif values.get("task") == "image-object-detection":
            if not values.get("column_mapping"):
                raise ValueError("column_mapping is required for image-object-detection")
            if not values.get("column_mapping").get("image_column"):
                raise ValueError("image_column is required for image-object-detection")
            if not values.get("column_mapping").get("objects_column"):
                raise ValueError("objects_column is required for image-object-detection")
            values["column_mapping"] = ObjectDetectionColumnMapping(**values["column_mapping"])
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_params(cls, values):
        if values.get("task") == "llm:sft":
            values["params"] = LLMSFTTrainingParamsAPI(**values["params"])
        elif values.get("task") == "llm:dpo":
            values["params"] = LLMDPOTrainingParamsAPI(**values["params"])
        elif values.get("task") == "llm:orpo":
            values["params"] = LLMORPOTrainingParamsAPI(**values["params"])
        elif values.get("task") == "llm:generic":
            values["params"] = LLMGenericTrainingParamsAPI(**values["params"])
        elif values.get("task") == "llm:reward":
            values["params"] = LLMRewardTrainingParamsAPI(**values["params"])
        elif values.get("task") == "seq2seq":
            values["params"] = Seq2SeqParamsAPI(**values["params"])
        elif values.get("task") == "image-classification":
            values["params"] = ImageClassificationParamsAPI(**values["params"])
        elif values.get("task") == "tabular-classification":
            values["params"] = TabularClassificationParamsAPI(**values["params"])
        elif values.get("task") == "tabular-regression":
            values["params"] = TabularRegressionParamsAPI(**values["params"])
        elif values.get("task") == "text-classification":
            values["params"] = TextClassificationParamsAPI(**values["params"])
        elif values.get("task") == "text-regression":
            values["params"] = TextRegressionParamsAPI(**values["params"])
        elif values.get("task") == "token-classification":
            values["params"] = TokenClassificationParamsAPI(**values["params"])
        elif values.get("task").startswith("st:"):
            values["params"] = SentenceTransformersParamsAPI(**values["params"])
        elif values.get("task") == "image-regression":
            values["params"] = ImageRegressionParamsAPI(**values["params"])
        elif values.get("task").startswith("vlm:"):
            values["params"] = VLMTrainingParamsAPI(**values["params"])
        elif values.get("task") == "extractive-question-answering":
            values["params"] = ExtractiveQuestionAnsweringParamsAPI(**values["params"])
        elif values.get("task") == "image-object-detection":
            values["params"] = ObjectDetectionParamsAPI(**values["params"])
        return values


class JobIDModel(BaseModel):
    jid: str


api_router = APIRouter()


def api_auth(request: Request):
    """
    Authenticates the API request using a Bearer token.

    Args:
        request (Request): The incoming HTTP request object.

    Returns:
        str: The verified Bearer token if authentication is successful.

    Raises:
        HTTPException: If the token is invalid, expired, or missing.
    """
    authorization = request.headers.get("Authorization")
    if authorization:
        schema, _, token = authorization.partition(" ")
        if schema.lower() == "bearer":
            token = token.strip()
            try:
                _ = token_verification(token=token)
                return token
            except Exception as e:
                logger.error(f"Failed to verify token: {e}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token: Bearer",
                )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
    )


@api_router.post("/create_project", response_class=JSONResponse)
async def api_create_project(project: APICreateProjectModel, token: bool = Depends(api_auth)):
    """
    Asynchronously creates a new project based on the provided parameters.

    Args:
        project (APICreateProjectModel): The model containing the project details and parameters.
        token (bool, optional): The authentication token. Defaults to Depends(api_auth).

    Returns:
        dict: A dictionary containing a success message, the job ID of the created project, and a success status.

    Raises:
        HTTPException: If there is an error during project creation.

    Notes:
        - The function determines the hardware type based on the project hardware attribute.
        - It logs the provided parameters and column mapping.
        - It sets the appropriate parameters based on the task type.
        - It updates the parameters with the provided ones and creates an AppParams instance.
        - The function then creates an AutoTrainProject instance and initiates the project creation process.
    """
    provided_params = project.params.model_dump()
    if project.hardware == "local":
        hardware = "local-ui"  # local-ui has wait=False
    else:
        hardware = project.hardware

    logger.info(provided_params)
    logger.info(project.column_mapping)

    task = project.task
    if task.startswith("llm"):
        params = PARAMS["llm"]
        trainer = task.split(":")[1]
        params.update({"trainer": trainer})
    elif task.startswith("st:"):
        params = PARAMS["st"]
        trainer = task.split(":")[1]
        params.update({"trainer": trainer})
    elif task.startswith("vlm:"):
        params = PARAMS["vlm"]
        trainer = task.split(":")[1]
        params.update({"trainer": trainer})
    elif task.startswith("tabular"):
        params = PARAMS["tabular"]
    else:
        params = PARAMS[task]

    params.update(provided_params)

    app_params = AppParams(
        job_params_json=json.dumps(params),
        token=token,
        project_name=project.project_name,
        username=project.username,
        task=task,
        data_path=project.hub_dataset,
        base_model=project.base_model,
        column_mapping=project.column_mapping.model_dump() if project.column_mapping else None,
        using_hub_dataset=True,
        train_split=project.train_split,
        valid_split=project.valid_split,
        api=True,
    )
    params = app_params.munge()
    project = AutoTrainProject(params=params, backend=hardware)
    job_id = project.create()
    return {"message": "Project created", "job_id": job_id, "success": True}


@api_router.get("/version", response_class=JSONResponse)
async def api_version():
    """
    Returns the current version of the API.

    This asynchronous function retrieves the version of the API from the
    __version__ variable and returns it in a dictionary.

    Returns:
        dict: A dictionary containing the API version.
    """
    return {"version": __version__}


@api_router.post("/stop_training", response_class=JSONResponse)
async def api_stop_training(job: JobIDModel, token: bool = Depends(api_auth)):
    """
    Stops the training job with the given job ID.

    This asynchronous function pauses the training job identified by the provided job ID.
    It uses the Hugging Face API to pause the space associated with the job.

    Args:
        job (JobIDModel): The job model containing the job ID.
        token (bool, optional): The authentication token, provided by dependency injection.

    Returns:
        dict: A dictionary containing a message and a success flag. If the training job
        was successfully stopped, the message indicates success and the success flag is True.
        If there was an error, the message contains the error details and the success flag is False.

    Raises:
        Exception: If there is an error while attempting to stop the training job.
    """
    hf_api = HfApi(token=token)
    job_id = job.jid
    try:
        hf_api.pause_space(repo_id=job_id)
    except Exception as e:
        logger.error(f"Failed to stop training: {e}")
        return {"message": f"Failed to stop training for {job_id}: {e}", "success": False}
    return {"message": f"Training stopped for {job_id}", "success": True}


@api_router.post("/logs", response_class=JSONResponse)
async def api_logs(job: JobIDModel, token: bool = Depends(api_auth)):
    """
    Fetch logs for a given job.

    This endpoint retrieves logs for a specified job by its job ID. It first obtains a JWT token
    to authenticate the request and then fetches the logs from the Hugging Face API.

    Args:
        job (JobIDModel): The job model containing the job ID.
        token (bool, optional): Dependency injection for API authentication. Defaults to Depends(api_auth).

    Returns:
        JSONResponse: A JSON response containing the logs, success status, and a message.

    Raises:
        Exception: If there is an error fetching the logs, the exception message is returned in the response.
    """
    job_id = job.jid
    jwt_url = f"{constants.ENDPOINT}/api/spaces/{job_id}/jwt"
    response = get_session().get(jwt_url, headers=build_hf_headers(token=token))
    hf_raise_for_status(response)
    jwt_token = response.json()["token"]  # works for 24h (see "exp" field)

    # fetch the logs
    logs_url = f"https://api.hf.space/v1/{job_id}/logs/run"

    _logs = []
    try:
        with get_session().get(
            logs_url, headers=build_hf_headers(token=jwt_token), stream=True, timeout=3
        ) as response:
            hf_raise_for_status(response)
            for line in response.iter_lines():
                if not line.startswith(b"data: "):
                    continue
                line_data = line[len(b"data: ") :]
                try:
                    event = json.loads(line_data.decode())
                except json.JSONDecodeError:
                    continue  # ignore (for example, empty lines or `b': keep-alive'`)
                _logs.append((event["timestamp"], event["data"]))

        _logs = "\n".join([f"{timestamp}: {data}" for timestamp, data in _logs])
        return {"logs": _logs, "success": True, "message": "Logs fetched successfully"}
    except Exception as e:
        if "Read timed out" in str(e):
            _logs = "\n".join([f"{timestamp}: {data}" for timestamp, data in _logs])
            return {"logs": _logs, "success": True, "message": "Logs fetched successfully"}
        return {"logs": str(e), "success": False, "message": "Failed to fetch logs"}
