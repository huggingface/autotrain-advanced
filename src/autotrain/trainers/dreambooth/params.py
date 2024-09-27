from typing import Optional

from pydantic import Field

from autotrain.trainers.common import AutoTrainParams


class DreamBoothTrainingParams(AutoTrainParams):
    """
    DreamBoothTrainingParams

    Attributes:
        model (str): Name of the model to be used for training.
        vae_model (Optional[str]): Name of the VAE model to be used, if any.
        revision (Optional[str]): Specific model version to use.
        tokenizer (Optional[str]): Tokenizer to be used, if different from the model.
        image_path (str): Path to the training images.
        class_image_path (Optional[str]): Path to the class images.
        prompt (str): Prompt for the instance images.
        class_prompt (Optional[str]): Prompt for the class images.
        num_class_images (int): Number of class images to generate.
        class_labels_conditioning (Optional[str]): Conditioning labels for class images.
        prior_preservation (bool): Enable prior preservation during training.
        prior_loss_weight (float): Weight of the prior preservation loss.
        project_name (str): Name of the project for output directory.
        seed (int): Random seed for reproducibility.
        resolution (int): Resolution of the training images.
        center_crop (bool): Enable center cropping of images.
        train_text_encoder (bool): Enable training of the text encoder.
        batch_size (int): Batch size for training.
        sample_batch_size (int): Batch size for sampling.
        epochs (int): Number of training epochs.
        num_steps (int): Maximum number of training steps.
        checkpointing_steps (int): Steps interval for checkpointing.
        resume_from_checkpoint (Optional[str]): Path to resume training from a checkpoint.
        gradient_accumulation (int): Number of gradient accumulation steps.
        disable_gradient_checkpointing (bool): Disable gradient checkpointing.
        lr (float): Learning rate for training.
        scale_lr (bool): Enable scaling of the learning rate.
        scheduler (str): Type of learning rate scheduler.
        warmup_steps (int): Number of warmup steps for learning rate scheduler.
        num_cycles (int): Number of cycles for learning rate scheduler.
        lr_power (float): Power factor for learning rate scheduler.
        dataloader_num_workers (int): Number of workers for data loading.
        use_8bit_adam (bool): Enable use of 8-bit Adam optimizer.
        adam_beta1 (float): Beta1 parameter for Adam optimizer.
        adam_beta2 (float): Beta2 parameter for Adam optimizer.
        adam_weight_decay (float): Weight decay for Adam optimizer.
        adam_epsilon (float): Epsilon parameter for Adam optimizer.
        max_grad_norm (float): Maximum gradient norm for clipping.
        allow_tf32 (bool): Allow use of TF32 for training.
        prior_generation_precision (Optional[str]): Precision for prior generation.
        local_rank (int): Local rank for distributed training.
        xformers (bool): Enable xformers memory efficient attention.
        pre_compute_text_embeddings (bool): Pre-compute text embeddings before training.
        tokenizer_max_length (Optional[int]): Maximum length for tokenizer.
        text_encoder_use_attention_mask (bool): Use attention mask for text encoder.
        rank (int): Rank for distributed training.
        xl (bool): Enable XL model training.
        mixed_precision (Optional[str]): Enable mixed precision training.
        token (Optional[str]): Token for accessing the model hub.
        push_to_hub (bool): Enable pushing the model to the hub.
        username (Optional[str]): Username for the model hub.
        validation_prompt (Optional[str]): Prompt for validation images.
        num_validation_images (int): Number of validation images to generate.
        validation_epochs (int): Epoch interval for validation.
        checkpoints_total_limit (Optional[int]): Total limit for checkpoints.
        validation_images (Optional[str]): Path to validation images.
        logging (bool): Enable logging using TensorBoard.
    """

    model: str = Field(None, title="Name of the model to be used for training")
    vae_model: Optional[str] = Field(None, title="Name of the VAE model to be used, if any")
    revision: Optional[str] = Field(None, title="Specific model version to use")
    tokenizer: Optional[str] = Field(None, title="Tokenizer to be used, if different from the model")
    image_path: str = Field(None, title="Path to the training images")
    class_image_path: Optional[str] = Field(None, title="Path to the class images")
    prompt: str = Field(None, title="Prompt for the instance images")
    class_prompt: Optional[str] = Field(None, title="Prompt for the class images")
    num_class_images: int = Field(100, title="Number of class images to generate")
    class_labels_conditioning: Optional[str] = Field(None, title="Conditioning labels for class images")

    prior_preservation: bool = Field(False, title="Enable prior preservation during training")
    prior_loss_weight: float = Field(1.0, title="Weight of the prior preservation loss")

    project_name: str = Field("dreambooth-model", title="Name of the project for output directory")
    seed: int = Field(42, title="Random seed for reproducibility")
    resolution: int = Field(512, title="Resolution of the training images")
    center_crop: bool = Field(False, title="Enable center cropping of images")
    train_text_encoder: bool = Field(False, title="Enable training of the text encoder")
    batch_size: int = Field(4, title="Batch size for training")
    sample_batch_size: int = Field(4, title="Batch size for sampling")
    epochs: int = Field(1, title="Number of training epochs")
    num_steps: int = Field(None, title="Maximum number of training steps")
    checkpointing_steps: int = Field(500, title="Steps interval for checkpointing")
    resume_from_checkpoint: Optional[str] = Field(None, title="Path to resume training from a checkpoint")

    gradient_accumulation: int = Field(1, title="Number of gradient accumulation steps")
    disable_gradient_checkpointing: bool = Field(False, title="Disable gradient checkpointing")

    lr: float = Field(1e-4, title="Learning rate for training")
    scale_lr: bool = Field(False, title="Enable scaling of the learning rate")
    scheduler: str = Field("constant", title="Type of learning rate scheduler")
    warmup_steps: int = Field(0, title="Number of warmup steps for learning rate scheduler")
    num_cycles: int = Field(1, title="Number of cycles for learning rate scheduler")
    lr_power: float = Field(1.0, title="Power factor for learning rate scheduler")

    dataloader_num_workers: int = Field(0, title="Number of workers for data loading")
    use_8bit_adam: bool = Field(False, title="Enable use of 8-bit Adam optimizer")
    adam_beta1: float = Field(0.9, title="Beta1 parameter for Adam optimizer")
    adam_beta2: float = Field(0.999, title="Beta2 parameter for Adam optimizer")
    adam_weight_decay: float = Field(1e-2, title="Weight decay for Adam optimizer")
    adam_epsilon: float = Field(1e-8, title="Epsilon parameter for Adam optimizer")
    max_grad_norm: float = Field(1.0, title="Maximum gradient norm for clipping")

    allow_tf32: bool = Field(False, title="Allow use of TF32 for training")
    prior_generation_precision: Optional[str] = Field(None, title="Precision for prior generation")
    local_rank: int = Field(-1, title="Local rank for distributed training")
    xformers: bool = Field(False, title="Enable xformers memory efficient attention")
    pre_compute_text_embeddings: bool = Field(False, title="Pre-compute text embeddings before training")
    tokenizer_max_length: Optional[int] = Field(None, title="Maximum length for tokenizer")
    text_encoder_use_attention_mask: bool = Field(False, title="Use attention mask for text encoder")

    rank: int = Field(4, title="Rank for distributed training")
    xl: bool = Field(False, title="Enable XL model training")

    mixed_precision: Optional[str] = Field(None, title="Enable mixed precision training")

    token: Optional[str] = Field(None, title="Token for accessing the model hub")
    push_to_hub: bool = Field(False, title="Enable pushing the model to the hub")
    username: Optional[str] = Field(None, title="Username for the model hub")

    # disabled:
    validation_prompt: Optional[str] = Field(None, title="Prompt for validation images")
    num_validation_images: int = Field(4, title="Number of validation images to generate")
    validation_epochs: int = Field(50, title="Epoch interval for validation")
    checkpoints_total_limit: Optional[int] = Field(None, title="Total limit for checkpoints")
    validation_images: Optional[str] = Field(None, title="Path to validation images")

    logging: bool = Field(False, title="Enable logging using TensorBoard")
