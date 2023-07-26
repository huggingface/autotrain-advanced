from pydantic import BaseModel, Field


class DreamBoothTrainingParams(BaseModel):
    model: str = Field(None, title="Model name")
    revision: str = Field(None, title="Revision")
    tokenizer: str = Field(None, title="Tokenizer, if different from model")
    image_path: str = Field(None, title="Image path")
    class_image_path: str = Field(None, title="Class image path")
    prompt: str = Field(None, title="Instance prompt")
    class_prompt: str = Field(None, title="Class prompt")
    num_class_images: int = Field(100, title="Number of class images")
    class_labels_conditioning: str = Field(None, title="Class labels conditioning")

    prior_preservation: bool = Field(False, title="With prior preservation")
    prior_loss_weight: float = Field(1.0, title="Prior loss weight")

    output: str = Field("dreambooth-model", title="Output directory")
    seed: int = Field(42, title="Seed")
    resolution: int = Field(512, title="Resolution")
    center_crop: bool = Field(False, title="Center crop")
    train_text_encoder: bool = Field(False, title="Train text encoder")
    batch_size: int = Field(4, title="Train batch size")
    sample_batch_size: int = Field(4, title="Sample batch size")
    epochs: int = Field(1, title="Number of training epochs")
    num_steps: int = Field(None, title="Max train steps")
    checkpointing_steps: int = Field(500, title="Checkpointing steps")
    resume_from_checkpoint: str = Field(None, title="Resume from checkpoint")

    gradient_accumulation: int = Field(1, title="Gradient accumulation steps")
    gradient_checkpointing: bool = Field(False, title="Gradient checkpointing")

    lr: float = Field(5e-4, title="Learning rate")
    scale_lr: bool = Field(False, title="Scale learning rate")
    scheduler: str = Field("constant", title="Learning rate scheduler")
    warmup_steps: int = Field(0, title="Learning rate warmup steps")
    num_cycles: int = Field(1, title="Learning rate num cycles")
    lr_power: float = Field(1.0, title="Learning rate power")

    dataloader_num_workers: int = Field(0, title="Dataloader num workers")
    use_8bit_adam: bool = Field(False, title="Use 8bit adam")
    adam_beta1: float = Field(0.9, title="Adam beta 1")
    adam_beta2: float = Field(0.999, title="Adam beta 2")
    adam_weight_decay: float = Field(1e-2, title="Adam weight decay")
    adam_epsilon: float = Field(1e-8, title="Adam epsilon")
    max_grad_norm: float = Field(1.0, title="Max grad norm")

    allow_tf32: bool = Field(False, title="Allow TF32")
    prior_generation_precision: str = Field(None, title="Prior generation precision")
    local_rank: int = Field(-1, title="Local rank")
    xformers: bool = Field(False, title="Enable xformers memory efficient attention")
    pre_compute_text_embeddings: bool = Field(False, title="Pre compute text embeddings")
    tokenizer_max_length: int = Field(None, title="Tokenizer max length")
    text_encoder_use_attention_mask: bool = Field(False, title="Text encoder use attention mask")

    rank: int = Field(4, title="Rank")
    xl: bool = Field(False, title="XL")

    fp16: bool = Field(False, title="FP16")
    bf16: bool = Field(False, title="BF16")

    hub_token: str = Field(None, title="Hub token")
    hub_model_id: str = Field(None, title="Hub model id")
    push_to_hub: bool = Field(False, title="Push to hub")

    # disabled:
    validation_prompt: str = Field(None, title="Validation prompt")
    num_validation_images: int = Field(4, title="Number of validation images")
    validation_epochs: int = Field(50, title="Validation epochs")
    checkpoints_total_limit: int = Field(None, title="Checkpoints total limit")
    validation_images: str = Field(None, title="Validation images")

    logging: bool = Field(False, title="Logging using tensorboard")
