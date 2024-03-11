import argparse
import json
import os

from huggingface_hub import create_repo, snapshot_download, upload_folder

from autotrain.trainers.common import monitor, pause_space, remove_autotrain_data
from autotrain.trainers.dreambooth import utils
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams


def parse_args():
    # get training_config.json from the end user
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()


@monitor
def train(config):
    if isinstance(config, dict):
        config = DreamBoothTrainingParams(**config)
    config.prompt = str(config.prompt).strip()

    if config.model in utils.XL_MODELS:
        config.xl = True

    if config.repo_id is None and config.username is not None:
        config.repo_id = f"{config.username}/{config.project_name}"

    if config.project_name == "/tmp/model":
        snapshot_download(
            repo_id=config.image_path,
            local_dir=config.project_name,
            token=config.token,
            repo_type="dataset",
        )
        config.image_path = "/tmp/model/concept1/"
    if config.image_path == f"{config.project_name}/autotrain-data":
        config.image_path = os.path.join(config.image_path, "concept1")

    if config.xl:
        from autotrain.trainers.dreambooth.train_xl import main

        class Args:
            output_dir = config.project_name
            pretrained_model_name_or_path = config.model
            pretrained_vae_model_name_or_path = None
            revision = config.revision
            variant = None
            dataset_name = None
            dataset_config_name = None
            instance_data_dir = config.image_path
            cache_dir = None
            image_column = "image"
            caption_column = None
            repeats = 1
            class_data_dir = config.class_image_path
            instance_prompt = config.prompt
            class_prompt = config.class_prompt
            validation_prompt = None
            num_validation_images = 4
            validation_epochs = 50
            with_prior_preservation = config.prior_preservation
            prior_loss_weight = config.prior_loss_weight
            num_class_images = config.num_class_images
            seed = config.seed
            resolution = config.resolution
            crops_coords_top_left_h = 0
            crops_coords_top_left_w = 0
            center_crop = config.center_crop
            train_text_encoder = config.train_text_encoder
            train_batch_size = config.batch_size
            sample_batch_size = config.sample_batch_size
            num_train_epochs = config.epochs
            max_train_steps = config.num_steps
            checkpointing_steps = config.checkpointing_steps
            resume_from_checkpoint = config.resume_from_checkpoint
            max_grad_norm = config.max_grad_norm
            learning_rate = config.lr
            scale_lr = config.scale_lr
            lr_scheduler = config.scheduler
            lr_warmup_steps = config.warmup_steps
            lr_num_cycles = config.num_cycles
            lr_power = config.lr_power
            dataloader_num_workers = config.dataloader_num_workers
            optimizer = "adamw"
            use_8bit_adam = config.use_8bit_adam
            adam_beta1 = config.adam_beta1
            adam_beta2 = config.adam_beta2
            hub_token = config.token
            hub_model_id = config.repo_id
            logging_dir = os.path.join(config.project_name, "logs")
            allow_tf32 = config.allow_tf32
            report_to = "tensorboard" if config.logging else None
            mixed_precision = config.mixed_precision
            prior_generation_precision = config.prior_generation_precision
            local_rank = config.local_rank
            enable_xformers_memory_efficient_attention = config.xformers
            rank = config.rank
            validation_images = None
            tokenizer_max_length = config.tokenizer_max_length
            text_encoder_use_attention_mask = config.text_encoder_use_attention_mask
            gradient_accumulation_steps = config.gradient_accumulation
            gradient_checkpointing = not config.disable_gradient_checkpointing
            adam_weight_decay_text_encoder = 1e-3
            adam_weight_decay = 1e-4
            adam_epsilon = 1e-8
            prodigy_beta3 = None
            prodigy_decouple = True
            prodigy_use_bias_correction = True
            prodigy_safeguard_warmup = True
            snr_gamma = None
            text_encoder_lr = 5e-6

        _args = Args()
        main(_args)
    else:
        from autotrain.trainers.dreambooth.train import main

        # def parse_args(input_args=None):
        #     parser = argparse.ArgumentParser(description="Simple example of a training script.")
        #     parser.add_argument(
        #         "--pretrained_model_name_or_path",
        #         type=str,
        #         default=None,
        #         required=True,
        #         help="Path to pretrained model or model identifier from huggingface.co/models.",
        #     )
        #     parser.add_argument(
        #         "--revision",
        #         type=str,
        #         default=None,
        #         required=False,
        #         help="Revision of pretrained model identifier from huggingface.co/models.",
        #     )
        #     parser.add_argument(
        #         "--variant",
        #         type=str,
        #         default=None,
        #         help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
        #     )
        #     parser.add_argument(
        #         "--tokenizer_name",
        #         type=str,
        #         default=None,
        #         help="Pretrained tokenizer name or path if not the same as model_name",
        #     )
        #     parser.add_argument(
        #         "--instance_data_dir",
        #         type=str,
        #         default=None,
        #         required=True,
        #         help="A folder containing the training data of instance images.",
        #     )
        #     parser.add_argument(
        #         "--class_data_dir",
        #         type=str,
        #         default=None,
        #         required=False,
        #         help="A folder containing the training data of class images.",
        #     )
        #     parser.add_argument(
        #         "--instance_prompt",
        #         type=str,
        #         default=None,
        #         required=True,
        #         help="The prompt with identifier specifying the instance",
        #     )
        #     parser.add_argument(
        #         "--class_prompt",
        #         type=str,
        #         default=None,
        #         help="The prompt to specify images in the same class as provided instance images.",
        #     )
        #     parser.add_argument(
        #         "--validation_prompt",
        #         type=str,
        #         default=None,
        #         help="A prompt that is used during validation to verify that the model is learning.",
        #     )
        #     parser.add_argument(
        #         "--num_validation_images",
        #         type=int,
        #         default=4,
        #         help="Number of images that should be generated during validation with `validation_prompt`.",
        #     )
        #     parser.add_argument(
        #         "--validation_epochs",
        #         type=int,
        #         default=50,
        #         help=(
        #             "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
        #             " `args.validation_prompt` multiple times: `args.num_validation_images`."
        #         ),
        #     )
        #     parser.add_argument(
        #         "--with_prior_preservation",
        #         default=False,
        #         action="store_true",
        #         help="Flag to add prior preservation loss.",
        #     )
        #     parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
        #     parser.add_argument(
        #         "--num_class_images",
        #         type=int,
        #         default=100,
        #         help=(
        #             "Minimal class images for prior preservation loss. If there are not enough images already present in"
        #             " class_data_dir, additional images will be sampled with class_prompt."
        #         ),
        #     )
        #     parser.add_argument(
        #         "--output_dir",
        #         type=str,
        #         default="lora-dreambooth-model",
        #         help="The output directory where the model predictions and checkpoints will be written.",
        #     )
        #     parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
        #     parser.add_argument(
        #         "--resolution",
        #         type=int,
        #         default=512,
        #         help=(
        #             "The resolution for input images, all the images in the train/validation dataset will be resized to this"
        #             " resolution"
        #         ),
        #     )
        #     parser.add_argument(
        #         "--center_crop",
        #         default=False,
        #         action="store_true",
        #         help=(
        #             "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
        #             " cropped. The images will be resized to the resolution first before cropping."
        #         ),
        #     )
        #     parser.add_argument(
        #         "--train_text_encoder",
        #         action="store_true",
        #         help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
        #     )
        #     parser.add_argument(
        #         "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
        #     )
        #     parser.add_argument(
        #         "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
        #     )
        #     parser.add_argument("--num_train_epochs", type=int, default=1)
        #     parser.add_argument(
        #         "--max_train_steps",
        #         type=int,
        #         default=None,
        #         help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
        #     )
        #     parser.add_argument(
        #         "--checkpointing_steps",
        #         type=int,
        #         default=500,
        #         help=(
        #             "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
        #             " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
        #             " training using `--resume_from_checkpoint`."
        #         ),
        #     )
        #     parser.add_argument(
        #         "--checkpoints_total_limit",
        #         type=int,
        #         default=None,
        #         help=("Max number of checkpoints to store."),
        #     )
        #     parser.add_argument(
        #         "--resume_from_checkpoint",
        #         type=str,
        #         default=None,
        #         help=(
        #             "Whether training should be resumed from a previous checkpoint. Use a path saved by"
        #             ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        #         ),
        #     )
        #     parser.add_argument(
        #         "--gradient_accumulation_steps",
        #         type=int,
        #         default=1,
        #         help="Number of updates steps to accumulate before performing a backward/update pass.",
        #     )
        #     parser.add_argument(
        #         "--gradient_checkpointing",
        #         action="store_true",
        #         help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
        #     )
        #     parser.add_argument(
        #         "--learning_rate",
        #         type=float,
        #         default=5e-4,
        #         help="Initial learning rate (after the potential warmup period) to use.",
        #     )
        #     parser.add_argument(
        #         "--scale_lr",
        #         action="store_true",
        #         default=False,
        #         help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
        #     )
        #     parser.add_argument(
        #         "--lr_scheduler",
        #         type=str,
        #         default="constant",
        #         help=(
        #             'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        #             ' "constant", "constant_with_warmup"]'
        #         ),
        #     )
        #     parser.add_argument(
        #         "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
        #     )
        #     parser.add_argument(
        #         "--lr_num_cycles",
        #         type=int,
        #         default=1,
        #         help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
        #     )
        #     parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
        #     parser.add_argument(
        #         "--dataloader_num_workers",
        #         type=int,
        #         default=0,
        #         help=(
        #             "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        #         ),
        #     )
        #     parser.add_argument(
        #         "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
        #     )
        #     parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
        #     parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
        #     parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
        #     parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
        #     parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        #     parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
        #     parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
        #     parser.add_argument(
        #         "--hub_model_id",
        #         type=str,
        #         default=None,
        #         help="The name of the repository to keep in sync with the local `output_dir`.",
        #     )
        #     parser.add_argument(
        #         "--logging_dir",
        #         type=str,
        #         default="logs",
        #         help=(
        #             "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
        #             " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        #         ),
        #     )
        #     parser.add_argument(
        #         "--allow_tf32",
        #         action="store_true",
        #         help=(
        #             "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
        #             " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        #         ),
        #     )
        #     parser.add_argument(
        #         "--report_to",
        #         type=str,
        #         default="tensorboard",
        #         help=(
        #             'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
        #             ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        #         ),
        #     )
        #     parser.add_argument(
        #         "--mixed_precision",
        #         type=str,
        #         default=None,
        #         choices=["no", "fp16", "bf16"],
        #         help=(
        #             "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
        #             " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
        #             " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        #         ),
        #     )
        #     parser.add_argument(
        #         "--prior_generation_precision",
        #         type=str,
        #         default=None,
        #         choices=["no", "fp32", "fp16", "bf16"],
        #         help=(
        #             "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
        #             " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        #         ),
        #     )
        #     parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
        #     parser.add_argument(
        #         "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
        #     )
        #     parser.add_argument(
        #         "--pre_compute_text_embeddings",
        #         action="store_true",
        #         help="Whether or not to pre-compute text embeddings. If text embeddings are pre-computed, the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model. This is not compatible with `--train_text_encoder`.",
        #     )
        #     parser.add_argument(
        #         "--tokenizer_max_length",
        #         type=int,
        #         default=None,
        #         required=False,
        #         help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",
        #     )
        #     parser.add_argument(
        #         "--text_encoder_use_attention_mask",
        #         action="store_true",
        #         required=False,
        #         help="Whether to use attention mask for the text encoder",
        #     )
        #     parser.add_argument(
        #         "--validation_images",
        #         required=False,
        #         default=None,
        #         nargs="+",
        #         help="Optional set of images to use for validation. Used when the target pipeline takes an initial image as input such as when training image variation or superresolution.",
        #     )
        #     parser.add_argument(
        #         "--class_labels_conditioning",
        #         required=False,
        #         default=None,
        #         help="The optional `class_label` conditioning to pass to the unet, available values are `timesteps`.",
        #     )
        #     parser.add_argument(
        #         "--rank",
        #         type=int,
        #         default=4,
        #         help=("The dimension of the LoRA update matrices."),
        #     )
        #     if input_args is not None:
        #         args = parser.parse_args(input_args)
        #     else:
        #         args = parser.parse_args()
        #     env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        #     if env_local_rank != -1 and env_local_rank != args.local_rank:
        #         args.local_rank = env_local_rank
        #     if args.with_prior_preservation:
        #         if args.class_data_dir is None:
        #             raise ValueError("You must specify a data directory for class images.")
        #         if args.class_prompt is None:
        #             raise ValueError("You must specify prompt for class images.")
        #     else:
        #         # logger is not available yet
        #         if args.class_data_dir is not None:
        #             warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        #         if args.class_prompt is not None:
        #             warnings.warn("You need not use --class_prompt without --with_prior_preservation.")
        #     if args.train_text_encoder and args.pre_compute_text_embeddings:
        #         raise ValueError("`--train_text_encoder` cannot be used with `--pre_compute_text_embeddings`")
        #     return args

        class Args:
            pretrained_model_name_or_path = config.model
            revision = config.revision
            variant = None
            tokenizer_name = None
            instance_data_dir = config.image_path
            class_data_dir = config.class_image_path
            instance_prompt = config.prompt
            class_prompt = config.class_prompt
            validation_prompt = None
            num_validation_images = 4
            validation_epochs = 50
            with_prior_preservation = config.prior_preservation
            prior_loss_weight = config.prior_loss_weight
            num_class_images = config.num_class_images
            output_dir = config.project_name
            seed = config.seed
            resolution = config.resolution
            center_crop = config.center_crop
            train_text_encoder = config.train_text_encoder
            train_batch_size = config.batch_size
            sample_batch_size = config.sample_batch_size
            num_train_epochs = config.epochs
            max_train_steps = config.num_steps
            checkpointing_steps = config.checkpointing_steps
            resume_from_checkpoint = config.resume_from_checkpoint
            gradient_accumulation_steps = config.gradient_accumulation
            gradient_checkpointing = config.disable_gradient_checkpointing
            learning_rate = config.lr
            scale_lr = config.scale_lr
            lr_scheduler = config.scheduler
            lr_warmup_steps = config.warmup_steps
            lr_num_cycles = config.num_cycles
            lr_power = config.lr_power
            dataloader_num_workers = config.dataloader_num_workers
            use_8bit_adam = config.use_8bit_adam
            adam_beta1 = config.adam_beta1
            adam_beta2 = config.adam_beta2
            adam_weight_decay = config.adam_weight_decay
            adam_epsilon = config.adam_epsilon
            max_grad_norm = config.max_grad_norm
            allow_tf32 = config.allow_tf32
            prior_generation_precision = config.prior_generation_precision
            local_rank = config.local_rank
            enable_xformers_memory_efficient_attention = config.xformers
            pre_compute_text_embeddings = config.pre_compute_text_embeddings
            tokenizer_max_length = config.tokenizer_max_length
            text_encoder_use_attention_mask = config.text_encoder_use_attention_mask
            rank = config.rank
            mixed_precision = config.mixed_precision
            token = config.token
            repo_id = config.repo_id
            push_to_hub = config.push_to_hub
            username = config.username
            report_to = "tensorboard" if config.logging else None
            logging_dir = os.path.join(config.project_name, "logs")
            validation_images = None
            class_labels_conditioning = None

        _args = Args()
        main(_args)

    if os.path.exists(f"{config.project_name}/training_params.json"):
        training_params = json.load(open(f"{config.project_name}/training_params.json"))
        if "token" in training_params:
            training_params.pop("token")
            json.dump(
                training_params,
                open(f"{config.project_name}/training_params.json", "w"),
            )

    # add config.prompt as a text file in the output directory
    with open(f"{config.project_name}/prompt.txt", "w") as f:
        f.write(config.prompt)

    if config.push_to_hub:
        remove_autotrain_data(config)

        repo_id = create_repo(
            repo_id=config.repo_id,
            exist_ok=True,
            private=True,
            token=config.token,
        ).repo_id
        if config.xl:
            utils.save_model_card_xl(
                repo_id,
                base_model=config.model,
                train_text_encoder=config.train_text_encoder,
                instance_prompt=config.prompt,
                vae_path=config.vae_model,
                repo_folder=config.project_name,
            )
        else:
            utils.save_model_card(
                repo_id,
                base_model=config.model,
                train_text_encoder=config.train_text_encoder,
                instance_prompt=config.prompt,
                repo_folder=config.project_name,
            )

        upload_folder(
            repo_id=repo_id,
            folder_path=config.project_name,
            commit_message="End of training",
            ignore_patterns=["step_*", "epoch_*"],
            token=config.token,
        )

    pause_space(config)


if __name__ == "__main__":
    args = parse_args()
    training_config = json.load(open(args.training_config))
    config = DreamBoothTrainingParams(**training_config)
    train(config)
