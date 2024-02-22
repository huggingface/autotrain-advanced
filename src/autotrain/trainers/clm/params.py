from typing import List, Optional, Union

from pydantic import Field

from autotrain.trainers.common import AutoTrainParams


class LLMTrainingParams(AutoTrainParams):
    model: str = Field("gpt2", title="Model name")
    project_name: str = Field("project-name", title="Output directory")

    # data params
    data_path: str = Field("data", title="Data path")
    train_split: str = Field("train", title="Train data config")
    valid_split: Optional[str] = Field(None, title="Validation data config")
    add_eos_token: bool = Field(True, title="Add EOS token")
    block_size: Union[int, List[int]] = Field(-1, title="Block size")
    model_max_length: int = Field(2048, title="Model max length")
    padding: Optional[str] = Field(None, title="Padding side")

    # trainer params
    trainer: str = Field("default", title="Trainer type")
    use_flash_attention_2: bool = Field(False, title="Use flash attention 2")
    log: str = Field("none", title="Logging using experiment tracking")
    disable_gradient_checkpointing: bool = Field(False, title="Gradient checkpointing")
    logging_steps: int = Field(-1, title="Logging steps")
    evaluation_strategy: str = Field("epoch", title="Evaluation strategy")
    save_total_limit: int = Field(1, title="Save total limit")
    save_strategy: str = Field("epoch", title="Save strategy")
    auto_find_batch_size: bool = Field(False, title="Auto find batch size")
    mixed_precision: Optional[str] = Field(None, title="fp16, bf16, or None")
    lr: float = Field(3e-5, title="Learning rate")
    epochs: int = Field(1, title="Number of training epochs")
    batch_size: int = Field(2, title="Training batch size")
    warmup_ratio: float = Field(0.1, title="Warmup proportion")
    gradient_accumulation: int = Field(1, title="Gradient accumulation steps")
    optimizer: str = Field("adamw_torch", title="Optimizer")
    scheduler: str = Field("linear", title="Scheduler")
    weight_decay: float = Field(0.0, title="Weight decay")
    max_grad_norm: float = Field(1.0, title="Max gradient norm")
    seed: int = Field(42, title="Seed")
    chat_template: Optional[str] = Field(None, title="Chat template, one of: None, zephyr, chatml or tokenizer")

    # peft
    quantization: Optional[str] = Field(None, title="int4, int8, or None")
    target_modules: Optional[str] = Field("all-linear", title="Target modules")
    merge_adapter: bool = Field(False, title="Merge adapter")
    peft: bool = Field(False, title="Use PEFT")
    lora_r: int = Field(16, title="Lora r")
    lora_alpha: int = Field(32, title="Lora alpha")
    lora_dropout: float = Field(0.05, title="Lora dropout")

    # dpo
    model_ref: Optional[str] = Field(None, title="Reference, for DPO trainer")
    dpo_beta: float = Field(0.1, title="Beta for DPO trainer")

    # column mappings
    prompt_text_column: Optional[str] = Field(None, title="Prompt text column")
    text_column: str = Field("text", title="Text column")
    rejected_text_column: Optional[str] = Field(None, title="Rejected text column")

    # push to hub
    push_to_hub: bool = Field(False, title="Push to hub")
    repo_id: Optional[str] = Field(None, title="Repo id")
    username: Optional[str] = Field(None, title="Hugging Face Username")
    token: Optional[str] = Field(None, title="Huggingface token")
