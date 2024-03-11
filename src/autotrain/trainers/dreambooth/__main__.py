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

    if config.vae_model is not None:
        if config.vae_model.strip() == "":
            config.vae_model = None

    if config.xl:
        from autotrain.trainers.dreambooth.train_xl import main

        class Args:
            pretrained_model_name_or_path = config.model
            pretrained_vae_model_name_or_path = config.vae_model
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
            num_class_images = config.num_class_images
            output_dir = config.project_name
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
            checkpoints_total_limit = None
            resume_from_checkpoint = config.resume_from_checkpoint
            gradient_accumulation_steps = config.gradient_accumulation
            gradient_checkpointing = not config.disable_gradient_checkpointing
            learning_rate = config.lr
            text_encoder_lr = 5e-6
            scale_lr = config.scale_lr
            lr_scheduler = config.scheduler
            snr_gamma = None
            lr_warmup_steps = config.warmup_steps
            lr_num_cycles = config.num_cycles
            lr_power = config.lr_power
            dataloader_num_workers = config.dataloader_num_workers
            optimizer = "AdamW"
            use_8bit_adam = config.use_8bit_adam
            adam_beta1 = config.adam_beta1
            adam_beta2 = config.adam_beta2
            prodigy_beta3 = None
            prodigy_decouple = True
            adam_weight_decay = config.adam_weight_decay
            adam_weight_decay_text_encoder = 1e-3
            adam_epsilon = config.adam_epsilon
            prodigy_use_bias_correction = True
            prodigy_safeguard_warmup = True
            max_grad_norm = config.max_grad_norm
            push_to_hub = config.push_to_hub
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

        _args = Args()
        main(_args)
    else:
        from autotrain.trainers.dreambooth.train import main

        class Args:
            pretrained_model_name_or_path = config.model
            pretrained_vae_model_name_or_path = config.vae_model
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
            num_class_images = config.num_class_images
            output_dir = config.project_name
            seed = config.seed
            resolution = config.resolution
            center_crop = config.center_crop
            train_text_encoder = config.train_text_encoder
            train_batch_size = config.batch_size
            sample_batch_size = config.sample_batch_size
            max_train_steps = config.num_steps
            checkpointing_steps = config.checkpointing_steps
            checkpoints_total_limit = None
            resume_from_checkpoint = config.resume_from_checkpoint
            gradient_accumulation_steps = config.gradient_accumulation
            gradient_checkpointing = not config.disable_gradient_checkpointing
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
            push_to_hub = config.push_to_hub
            hub_token = config.token
            hub_model_id = config.repo_id
            logging_dir = os.path.join(config.project_name, "logs")
            allow_tf32 = config.allow_tf32
            report_to = "tensorboard" if config.logging else None
            mixed_precision = config.mixed_precision
            prior_generation_precision = config.prior_generation_precision
            local_rank = config.local_rank
            enable_xformers_memory_efficient_attention = config.xformers
            pre_compute_text_embeddings = config.pre_compute_text_embeddings
            tokenizer_max_length = config.tokenizer_max_length
            text_encoder_use_attention_mask = config.text_encoder_use_attention_mask
            validation_images = None
            class_labels_conditioning = config.class_labels_conditioning
            rank = config.rank

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
