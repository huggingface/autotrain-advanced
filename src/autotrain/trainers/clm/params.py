from typing import List, Optional, Union

from pydantic import Field

from autotrain.trainers.common import AutoTrainParams


class LLMTrainingParams(AutoTrainParams):
    """
    LLMTrainingParams: Parameters for training a language model using the autotrain library.

    Attributes:
        model (str): Model name to be used for training. Default is "gpt2".
        project_name (str): Name of the project and output directory. Default is "project-name".

        data_path (str): Path to the dataset. Default is "data".
        train_split (str): Configuration for the training data split. Default is "train".
        valid_split (Optional[str]): Configuration for the validation data split. Default is None.
        add_eos_token (bool): Whether to add an EOS token at the end of sequences. Default is True.
        block_size (Union[int, List[int]]): Size of the blocks for training, can be a single integer or a list of integers. Default is -1.
        model_max_length (int): Maximum length of the model input. Default is 2048.
        padding (Optional[str]): Side on which to pad sequences (left or right). Default is "right".

        trainer (str): Type of trainer to use. Default is "default".
        use_flash_attention_2 (bool): Whether to use flash attention version 2. Default is False.
        log (str): Logging method for experiment tracking. Default is "none".
        disable_gradient_checkpointing (bool): Whether to disable gradient checkpointing. Default is False.
        logging_steps (int): Number of steps between logging events. Default is -1.
        eval_strategy (str): Strategy for evaluation (e.g., 'epoch'). Default is "epoch".
        save_total_limit (int): Maximum number of checkpoints to keep. Default is 1.
        auto_find_batch_size (bool): Whether to automatically find the optimal batch size. Default is False.
        mixed_precision (Optional[str]): Type of mixed precision to use (e.g., 'fp16', 'bf16', or None). Default is None.
        lr (float): Learning rate for training. Default is 3e-5.
        epochs (int): Number of training epochs. Default is 1.
        batch_size (int): Batch size for training. Default is 2.
        warmup_ratio (float): Proportion of training to perform learning rate warmup. Default is 0.1.
        gradient_accumulation (int): Number of steps to accumulate gradients before updating. Default is 4.
        optimizer (str): Optimizer to use for training. Default is "adamw_torch".
        scheduler (str): Learning rate scheduler to use. Default is "linear".
        weight_decay (float): Weight decay to apply to the optimizer. Default is 0.0.
        max_grad_norm (float): Maximum norm for gradient clipping. Default is 1.0.
        seed (int): Random seed for reproducibility. Default is 42.
        chat_template (Optional[str]): Template for chat-based models, options include: None, zephyr, chatml, or tokenizer. Default is None.

        quantization (Optional[str]): Quantization method to use (e.g., 'int4', 'int8', or None). Default is "int4".
        target_modules (Optional[str]): Target modules for quantization or fine-tuning. Default is "all-linear".
        merge_adapter (bool): Whether to merge the adapter layers. Default is False.
        peft (bool): Whether to use Parameter-Efficient Fine-Tuning (PEFT). Default is False.
        lora_r (int): Rank of the LoRA matrices. Default is 16.
        lora_alpha (int): Alpha parameter for LoRA. Default is 32.
        lora_dropout (float): Dropout rate for LoRA. Default is 0.05.

        model_ref (Optional[str]): Reference model for DPO trainer. Default is None.
        dpo_beta (float): Beta parameter for DPO trainer. Default is 0.1.

        max_prompt_length (int): Maximum length of the prompt. Default is 128.
        max_completion_length (Optional[int]): Maximum length of the completion. Default is None.

        prompt_text_column (Optional[str]): Column name for the prompt text. Default is None.
        text_column (str): Column name for the text data. Default is "text".
        rejected_text_column (Optional[str]): Column name for the rejected text data. Default is None.

        push_to_hub (bool): Whether to push the model to the Hugging Face Hub. Default is False.
        username (Optional[str]): Hugging Face username for authentication. Default is None.
        token (Optional[str]): Hugging Face token for authentication. Default is None.

        unsloth (bool): Whether to use the unsloth library. Default is False.
        distributed_backend (Optional[str]): Backend to use for distributed training. Default is None.
    """

    model: str = Field("gpt2", title="Model name to be used for training")
    project_name: str = Field("project-name", title="Name of the project and output directory")

    # data params
    data_path: str = Field("data", title="Path to the dataset")
    train_split: str = Field("train", title="Configuration for the training data split")
    valid_split: Optional[str] = Field(None, title="Configuration for the validation data split")
    add_eos_token: bool = Field(True, title="Whether to add an EOS token at the end of sequences")
    block_size: Union[int, List[int]] = Field(
        -1, title="Size of the blocks for training, can be a single integer or a list of integers"
    )
    model_max_length: int = Field(2048, title="Maximum length of the model input")
    padding: Optional[str] = Field("right", title="Side on which to pad sequences (left or right)")

    # trainer params
    trainer: str = Field("default", title="Type of trainer to use")
    use_flash_attention_2: bool = Field(False, title="Whether to use flash attention version 2")
    log: str = Field("none", title="Logging method for experiment tracking")
    disable_gradient_checkpointing: bool = Field(False, title="Whether to disable gradient checkpointing")
    logging_steps: int = Field(-1, title="Number of steps between logging events")
    eval_strategy: str = Field("epoch", title="Strategy for evaluation (e.g., 'epoch')")
    save_total_limit: int = Field(1, title="Maximum number of checkpoints to keep")
    auto_find_batch_size: bool = Field(False, title="Whether to automatically find the optimal batch size")
    mixed_precision: Optional[str] = Field(
        None, title="Type of mixed precision to use (e.g., 'fp16', 'bf16', or None)"
    )
    lr: float = Field(3e-5, title="Learning rate for training")
    epochs: int = Field(1, title="Number of training epochs")
    batch_size: int = Field(2, title="Batch size for training")
    warmup_ratio: float = Field(0.1, title="Proportion of training to perform learning rate warmup")
    gradient_accumulation: int = Field(4, title="Number of steps to accumulate gradients before updating")
    optimizer: str = Field("adamw_torch", title="Optimizer to use for training")
    scheduler: str = Field("linear", title="Learning rate scheduler to use")
    weight_decay: float = Field(0.0, title="Weight decay to apply to the optimizer")
    max_grad_norm: float = Field(1.0, title="Maximum norm for gradient clipping")
    seed: int = Field(42, title="Random seed for reproducibility")
    chat_template: Optional[str] = Field(
        None, title="Template for chat-based models, options include: None, zephyr, chatml, or tokenizer"
    )

    # peft
    quantization: Optional[str] = Field("int4", title="Quantization method to use (e.g., 'int4', 'int8', or None)")
    target_modules: Optional[str] = Field("all-linear", title="Target modules for quantization or fine-tuning")
    merge_adapter: bool = Field(False, title="Whether to merge the adapter layers")
    peft: bool = Field(False, title="Whether to use Parameter-Efficient Fine-Tuning (PEFT)")
    lora_r: int = Field(16, title="Rank of the LoRA matrices")
    lora_alpha: int = Field(32, title="Alpha parameter for LoRA")
    lora_dropout: float = Field(0.05, title="Dropout rate for LoRA")

    # dpo
    model_ref: Optional[str] = Field(None, title="Reference model for DPO trainer")
    dpo_beta: float = Field(0.1, title="Beta parameter for DPO trainer")

    # orpo + dpo
    max_prompt_length: int = Field(128, title="Maximum length of the prompt")
    max_completion_length: Optional[int] = Field(None, title="Maximum length of the completion")

    # column mappings
    prompt_text_column: Optional[str] = Field(None, title="Column name for the prompt text")
    text_column: str = Field("text", title="Column name for the text data")
    rejected_text_column: Optional[str] = Field(None, title="Column name for the rejected text data")

    # push to hub
    push_to_hub: bool = Field(False, title="Whether to push the model to the Hugging Face Hub")
    username: Optional[str] = Field(None, title="Hugging Face username for authentication")
    token: Optional[str] = Field(None, title="Hugging Face token for authentication")

    # unsloth
    unsloth: bool = Field(False, title="Whether to use the unsloth library")
    distributed_backend: Optional[str] = Field(None, title="Backend to use for distributed training")
