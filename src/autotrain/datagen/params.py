from typing import Optional

from pydantic import Field

from autotrain.trainers.common import AutoTrainParams


class AutoTrainGenParams(AutoTrainParams):
    gen_model: str = Field("meta-llama/Meta-Llama-3.1-8B-Instruct", title="The model to be used for generation.")
    project_name: str = Field("autotrain-datagen", title="Name of the project.")
    prompt: str = Field(None, title="Prompt to be used for text generation.")
    task: str = Field(None, title="Task type, e.g., text-classification, summarization.")
    token: Optional[str] = Field(None, title="Authentication token for accessing the model.")
    training_config: Optional[str] = Field(None, title="Path to the training configuration file.")
    valid_size: Optional[float] = Field(0.2, title="Validation set size as a fraction of the total dataset.")
    username: Optional[str] = Field(None, title="Username of the person running the training.")
    push_to_hub: Optional[bool] = Field(True, title="Whether to push the model to Hugging Face Hub.")
    backend: Optional[str] = Field("huggingface", title="Backend to be used, e.g., huggingface, local.")
    api: Optional[str] = Field(None, title="API endpoint to be used.")
    api_key: Optional[str] = Field(None, title="API key for authentication.")
    min_samples: Optional[int] = Field(200, title="Minimum number of samples required for training.")
    # text specific
    min_words: Optional[int] = Field(25, title="Minimum number of words in the generated text.")
