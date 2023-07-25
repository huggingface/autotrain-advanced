from argparse import ArgumentParser

from loguru import logger

from autotrain.params import DreamBoothTrainingParams

from ..trainers.dream import train as train_dreambooth
from . import BaseAutoTrainCommand


def run_dreambooth_command_factory(args):
    return RunAutoTrainDreamboothCommand(
        args.model,
        args.revision,
        args.tokenizer,
        args.image_path,
        args.class_image_path,
        args.instance_prompt,
        args.class_prompt,
        args.validation_prompt,
        args.num_validation_images,
        args.validation_epochs,
        args.prior_preservation,
        args.prior_loss_weight,
        args.num_class_images,
        args.output_dir,
        args.seed,
        args.resolution,
        args.center_crop,
        args.train_text_encoder,
        args.train_batch_size,
        args.sample_batch_size,
        args.num_train_epochs,
        args.max_train_steps,
        args.checkpointing_steps,
        args.checkpoints_total_limit,
        args.resume_from_checkpoint,
        args.gradient_accumulation_steps,
        args.gradient_checkpointing,
        args.learning_rate,
        args.scale_lr,
        args.lr_scheduler,
        args.lr_warmup_steps,
        args.lr_num_cycles,
        args.lr_power,
        args.dataloader_num_workers,
        args.use_8bit_adam,
        args.adam_beta1,
        args.adam_beta2,
        args.adam_weight_decay,
        args.adam_epsilon,
        args.max_grad_norm,
        args.push_to_hub,
        args.hub_token,
        args.hub_model_id,
        args.logging_dir,
        args.allow_tf32,
        args.report_to,
        args.mixed_precision,
        args.prior_generation_precision,
        args.local_rank,
        args.enable_xformers_memory_efficient_attention,
        args.pre_compute_text_embeddings,
        args.tokenizer_max_length,
        args.text_encoder_use_attention_mask,
        args.validation_images,
        args.class_labels_conditioning,
        args.rank,
    )


class RunAutoTrainDreamboothCommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        run_dreambooth_parser = parser.add_parser(
            "dreambooth",
            description="✨ Run AutoTrain DreamBooth Training",
        )
        run_dreambooth_parser.add_argument(
            "--model",
            help="Model to use for training",
            required=True,
            default="stabilityai/stable-diffusion-xl-base-0.9",
        )
        run_dreambooth_parser.add_argument(
            "--revision",
            help="Revision of model to use for training",
            required=False,
            default=None,
        )
        run_dreambooth_parser.add_argument(
            "--tokenizer",
            help="Tokenizer to use for training, if different from model",
            required=False,
            default=None,
        )
        run_dreambooth_parser.add_argument(
            "--image-path",
            help="Path to concept images",
            required=True,
            default=None,
        )
        run_dreambooth_parser.add_argument(
            "--class-image-path",
            help="Path to class images",
            required=False,
            default=None,
        )
        run_dreambooth_parser.add_argument(
            "--instance-prompt",
            help="Instance prompt",
            required=True,
            default=None,
        )
        run_dreambooth_parser.add_argument(
            "--class-prompt",
            help="Class prompt",
            required=False,
            default=None,
        )
        run_dreambooth_parser.add_argument(
            "--validation-prompt",
            help="Validation prompt",
            required=False,
            default=None,
        )
        run_dreambooth_parser.add_argument(
            "--num-validation-images",
            help="Number of validation images",
            required=False,
            default=4,
        )
        run_dreambooth_parser.add_argument(
            "--validation-epochs",
            help="Validation epochs",
            required=False,
            default=50,
        )
        run_dreambooth_parser.add_argument(
            "--prior-preservation",
            help="Enable prior preservation",
            required=False,
            action="store_true",
        )
        run_dreambooth_parser.add_argument(
            "--prior-loss-weight",
            help="Prior loss weight",
            required=False,
            default=1.0,
        )
        run_dreambooth_parser.add_argument(
            "--num-class-images",
            help="Number of class images",
            required=False,
            default=100,
        )
        run_dreambooth_parser.add_argument(
            "--output-dir",
            help="Output directory",
            required=False,
            default="autotrain-dreambooth-model",
        )
        run_dreambooth_parser.add_argument(
            "--seed",
            help="Seed",
            required=False,
            default=42,
        )
        run_dreambooth_parser.add_argument(
            "--resolution",
            help="Resolution",
            required=False,
            default=512,
        )
        run_dreambooth_parser.add_argument(
            "--center-crop",
            help="Center crop",
            required=False,
            action="store_true",
        )
        run_dreambooth_parser.add_argument(
            "--train-text-encoder",
            help="Train text encoder",
            required=False,
            action="store_true",
        )
        run_dreambooth_parser.add_argument(
            "--train-batch-size",
            help="Train batch size",
            required=False,
            default=4,
        )
        run_dreambooth_parser.add_argument(
            "--sample-batch-size",
            help="Sample batch size",
            required=False,
            default=4,
        )
        run_dreambooth_parser.add_argument(
            "--num-train-epochs",
            help="Number of training epochs",
            required=False,
            default=1,
        )
        run_dreambooth_parser.add_argument(
            "--max-train-steps",
            help="Max train steps",
            required=False,
            default=None,
        )
        run_dreambooth_parser.add_argument(
            "--checkpointing-steps",
            help="Checkpointing steps",
            required=False,
            default=500,
        )
        run_dreambooth_parser.add_argument(
            "--checkpoints-total-limit",
            help="Checkpoints total limit",
            required=False,
            default=None,
        )
        run_dreambooth_parser.add_argument(
            "--resume-from-checkpoint",
            help="Resume from checkpoint",
            required=False,
            default=None,
        )
        run_dreambooth_parser.add_argument(
            "--gradient-accumulation-steps",
            help="Gradient accumulation steps",
            required=False,
            default=1,
        )
        run_dreambooth_parser.add_argument(
            "--gradient-checkpointing",
            help="Gradient checkpointing",
            required=False,
            action="store_true",
        )
        run_dreambooth_parser.add_argument(
            "--learning-rate",
            help="Learning rate",
            required=False,
            default=5e-4,
        )
        run_dreambooth_parser.add_argument(
            "--scale-lr",
            help="Scale learning rate",
            required=False,
            action="store_true",
        )
        run_dreambooth_parser.add_argument(
            "--lr-scheduler",
            help="Learning rate scheduler",
            required=False,
            default="constant",
        )
        run_dreambooth_parser.add_argument(
            "--lr-warmup-steps",
            help="Learning rate warmup steps",
            required=False,
            default=500,
        )
        run_dreambooth_parser.add_argument(
            "--lr-num-cycles",
            help="Learning rate num cycles",
            required=False,
            default=1,
        )
        run_dreambooth_parser.add_argument(
            "--lr-power",
            help="Learning rate power",
            required=False,
            default=1.0,
        )
        run_dreambooth_parser.add_argument(
            "--dataloader-num-workers",
            help="Dataloader num workers",
            required=False,
            default=0,
        )
        run_dreambooth_parser.add_argument(
            "--use-8bit-adam",
            help="Use 8bit adam",
            required=False,
            action="store_true",
        )
        run_dreambooth_parser.add_argument(
            "--adam-beta1",
            help="Adam beta 1",
            required=False,
            default=0.9,
        )
        run_dreambooth_parser.add_argument(
            "--adam-beta2",
            help="Adam beta 2",
            required=False,
            default=0.999,
        )
        run_dreambooth_parser.add_argument(
            "--adam-weight-decay",
            help="Adam weight decay",
            required=False,
            default=1e-2,
        )
        run_dreambooth_parser.add_argument(
            "--adam-epsilon",
            help="Adam epsilon",
            required=False,
            default=1e-8,
        )
        run_dreambooth_parser.add_argument(
            "--max-grad-norm",
            help="Max grad norm",
            required=False,
            default=1.0,
        )
        run_dreambooth_parser.add_argument(
            "--push-to-hub",
            help="Push to hub",
            required=False,
            action="store_true",
        )
        run_dreambooth_parser.add_argument(
            "--hub-token",
            help="Hub token",
            required=False,
            default=None,
        )
        run_dreambooth_parser.add_argument(
            "--hub-model-id",
            help="Hub model id",
            required=False,
            default=None,
        )
        run_dreambooth_parser.add_argument(
            "--logging-dir",
            help="Logging directory",
            required=False,
            default="logs",
        )
        run_dreambooth_parser.add_argument(
            "--allow-tf32",
            help="Allow TF32",
            required=False,
            action="store_true",
        )
        run_dreambooth_parser.add_argument(
            "--report-to",
            help="Report to",
            required=False,
            default="tensorboard",
        )
        run_dreambooth_parser.add_argument(
            "--mixed-precision",
            help="Mixed precision",
            required=False,
            default=None,
        )
        run_dreambooth_parser.add_argument(
            "--prior-generation-precision",
            help="Prior generation precision",
            required=False,
            default=None,
        )
        run_dreambooth_parser.add_argument(
            "--local-rank",
            help="Local rank",
            required=False,
            default=-1,
        )
        run_dreambooth_parser.add_argument(
            "--enable-xformers-memory-efficient-attention",
            help="Enable xformers memory efficient attention",
            required=False,
            action="store_true",
        )
        run_dreambooth_parser.add_argument(
            "--pre-compute-text-embeddings",
            help="Pre compute text embeddings",
            required=False,
            action="store_true",
        )
        run_dreambooth_parser.add_argument(
            "--tokenizer-max-length",
            help="Tokenizer max length",
            required=False,
            default=None,
        )
        run_dreambooth_parser.add_argument(
            "--text-encoder-use-attention-mask",
            help="Text encoder use attention mask",
            required=False,
            action="store_true",
        )
        run_dreambooth_parser.add_argument(
            "--validation-images",
            help="Validation images",
            required=False,
            default=None,
        )
        run_dreambooth_parser.add_argument(
            "--class-labels-conditioning",
            help="Class labels conditioning",
            required=False,
            default=None,
        )
        run_dreambooth_parser.add_argument(
            "--rank",
            help="Rank",
            required=False,
            default=4,
        )
        run_dreambooth_parser.set_defaults(func=run_dreambooth_command_factory)

    def __init__(
        self,
        model,
        revision,
        tokenizer,
        image_path,
        class_image_path,
        instance_prompt,
        class_prompt,
        validation_prompt,
        num_validation_images,
        validation_epochs,
        with_prior_preservation,
        prior_loss_weight,
        num_class_images,
        output_dir,
        seed,
        resolution,
        center_crop,
        train_text_encoder,
        train_batch_size,
        sample_batch_size,
        num_train_epochs,
        max_train_steps,
        checkpointing_steps,
        checkpoints_total_limit,
        resume_from_checkpoint,
        gradient_accumulation_steps,
        gradient_checkpointing,
        learning_rate,
        scale_lr,
        lr_scheduler,
        lr_warmup_steps,
        lr_num_cycles,
        lr_power,
        dataloader_num_workers,
        use_8bit_adam,
        adam_beta1,
        adam_beta2,
        adam_weight_decay,
        adam_epsilon,
        max_grad_norm,
        push_to_hub,
        hub_token,
        hub_model_id,
        logging_dir,
        allow_tf32,
        report_to,
        mixed_precision,
        prior_generation_precision,
        local_rank,
        enable_xformers_memory_efficient_attention,
        pre_compute_text_embeddings,
        tokenizer_max_length,
        text_encoder_use_attention_mask,
        validation_images,
        class_labels_conditioning,
        rank,
    ):
        self.model = model
        self.revision = revision
        self.tokenizer = tokenizer
        self.image_path = image_path
        self.class_image_path = class_image_path
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.validation_prompt = validation_prompt
        self.num_validation_images = num_validation_images
        self.validation_epochs = validation_epochs
        self.with_prior_preservation = with_prior_preservation
        self.prior_loss_weight = prior_loss_weight
        self.num_class_images = num_class_images
        self.output_dir = output_dir
        self.seed = seed
        self.resolution = resolution
        self.center_crop = center_crop
        self.train_text_encoder = train_text_encoder
        self.train_batch_size = train_batch_size
        self.sample_batch_size = sample_batch_size
        self.num_train_epochs = num_train_epochs
        self.max_train_steps = max_train_steps
        self.checkpointing_steps = checkpointing_steps
        self.checkpoints_total_limit = checkpoints_total_limit
        self.resume_from_checkpoint = resume_from_checkpoint
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.learning_rate = learning_rate
        self.scale_lr = scale_lr
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_num_cycles = lr_num_cycles
        self.lr_power = lr_power
        self.dataloader_num_workers = dataloader_num_workers
        self.use_8bit_adam = use_8bit_adam
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_weight_decay = adam_weight_decay
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        self.push_to_hub = push_to_hub
        self.hub_token = hub_token
        self.hub_model_id = hub_model_id
        self.logging_dir = logging_dir
        self.allow_tf32 = allow_tf32
        self.report_to = report_to
        self.mixed_precision = mixed_precision
        self.prior_generation_precision = prior_generation_precision
        self.local_rank = local_rank
        self.enable_xformers_memory_efficient_attention = enable_xformers_memory_efficient_attention
        self.pre_compute_text_embeddings = pre_compute_text_embeddings
        self.tokenizer_max_length = tokenizer_max_length
        self.text_encoder_use_attention_mask = text_encoder_use_attention_mask
        self.validation_images = validation_images
        self.class_labels_conditioning = class_labels_conditioning
        self.rank = rank

    def run(self):
        logger.info("Running DreamBooth Training")
        params = DreamBoothTrainingParams(
            model_name=self.model,
            revision=self.revision,
            tokenizer=self.tokenizer,
            image_path=self.image_path,
            class_image_path=self.class_image_path,
            instance_prompt=self.instance_prompt,
            class_prompt=self.class_prompt,
            validation_prompt=self.validation_prompt,
            num_validation_images=self.num_validation_images,
            validation_epochs=self.validation_epochs,
            with_prior_preservation=self.with_prior_preservation,
            prior_loss_weight=self.prior_loss_weight,
            num_class_images=self.num_class_images,
            output_dir=self.output_dir,
            seed=self.seed,
            resolution=self.resolution,
            center_crop=self.center_crop,
            train_text_encoder=self.train_text_encoder,
            train_batch_size=self.train_batch_size,
            sample_batch_size=self.sample_batch_size,
            num_train_epochs=self.num_train_epochs,
            max_train_steps=self.max_train_steps,
            checkpointing_steps=self.checkpointing_steps,
            checkpoints_total_limit=self.checkpoints_total_limit,
            resume_from_checkpoint=self.resume_from_checkpoint,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            gradient_checkpointing=self.gradient_checkpointing,
            learning_rate=self.learning_rate,
            scale_lr=self.scale_lr,
            lr_scheduler=self.lr_scheduler,
            lr_warmup_steps=self.lr_warmup_steps,
            lr_num_cycles=self.lr_num_cycles,
            lr_power=self.lr_power,
            dataloader_num_workers=self.dataloader_num_workers,
            use_8bit_adam=self.use_8bit_adam,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            adam_weight_decay=self.adam_weight_decay,
            adam_epsilon=self.adam_epsilon,
            max_grad_norm=self.max_grad_norm,
            push_to_hub=self.push_to_hub,
            hub_token=self.hub_token,
            hub_model_id=self.hub_model_id,
            logging_dir=self.logging_dir,
            allow_tf32=self.allow_tf32,
            report_to=self.report_to,
            mixed_precision=self.mixed_precision,
            prior_generation_precision=self.prior_generation_precision,
            local_rank=self.local_rank,
            enable_xformers_memory_efficient_attention=self.enable_xformers_memory_efficient_attention,
            pre_compute_text_embeddings=self.pre_compute_text_embeddings,
            tokenizer_max_length=self.tokenizer_max_length,
            text_encoder_use_attention_mask=self.text_encoder_use_attention_mask,
            validation_images=self.validation_images,
            class_labels_conditioning=self.class_labels_conditioning,
            rank=self.rank,
        )
        train_dreambooth(params)
