import os
import subprocess
import sys
from argparse import ArgumentParser

import torch

from autotrain import logger

from . import BaseAutoTrainCommand


def run_llm_command_factory(args):
    return RunAutoTrainLLMCommand(args)


class RunAutoTrainLLMCommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        arg_list = [
            {
                "arg": "--train",
                "help": "Train the model",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--deploy",
                "help": "Deploy the model",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--inference",
                "help": "Run inference",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--data_path",
                "help": "Train dataset to use",
                "required": False,
                "type": str,
                "alias": ["--data-path"],
            },
            {
                "arg": "--train_split",
                "help": "Test dataset split to use",
                "required": False,
                "type": str,
                "default": "train",
                "alias": ["--train-split"],
            },
            {
                "arg": "--valid_split",
                "help": "Validation dataset split to use",
                "required": False,
                "type": str,
                "default": None,
                "alias": ["--valid-split"],
            },
            {
                "arg": "--text_column",
                "help": "Text column to use",
                "required": False,
                "type": str,
                "default": "text",
                "alias": ["--text-column"],
            },
            {
                "arg": "--rejected_text_column",
                "help": "Rejected text column to use",
                "required": False,
                "type": str,
                "default": "rejected",
                "alias": ["--rejected-text-column"],
            },
            {
                "arg": "--model",
                "help": "Model to use",
                "required": False,
                "type": str,
            },
            {
                "arg": "--learning_rate",
                "help": "Learning rate to use",
                "required": False,
                "type": float,
                "default": 3e-5,
                "alias": ["--lr", "--learning-rate"],
            },
            {
                "arg": "--num_train_epochs",
                "help": "Number of training epochs to use",
                "required": False,
                "type": int,
                "default": 1,
                "alias": ["--epochs"],
            },
            {
                "arg": "--train_batch_size",
                "help": "Training batch size to use",
                "required": False,
                "type": int,
                "default": 2,
                "alias": ["--train-batch-size", "--batch-size"],
            },
            {
                "arg": "--warmup_ratio",
                "help": "Warmup proportion to use",
                "required": False,
                "type": float,
                "default": 0.1,
                "alias": ["--warmup-ratio"],
            },
            {
                "arg": "--gradient_accumulation_steps",
                "help": "Gradient accumulation steps to use",
                "required": False,
                "type": int,
                "default": 1,
                "alias": ["--gradient-accumulation-steps", "--gradient-accumulation"],
            },
            {
                "arg": "--optimizer",
                "help": "Optimizer to use",
                "required": False,
                "type": str,
                "default": "adamw_torch",
            },
            {
                "arg": "--scheduler",
                "help": "Scheduler to use",
                "required": False,
                "type": str,
                "default": "linear",
            },
            {
                "arg": "--weight_decay",
                "help": "Weight decay to use",
                "required": False,
                "type": float,
                "default": 0.0,
                "alias": ["--weight-decay"],
            },
            {
                "arg": "--max_grad_norm",
                "help": "Max gradient norm to use",
                "required": False,
                "type": float,
                "default": 1.0,
                "alias": ["--max-grad-norm"],
            },
            {
                "arg": "--seed",
                "help": "Seed to use",
                "required": False,
                "type": int,
                "default": 42,
            },
            {
                "arg": "--add_eos_token",
                "help": "Add EOS token to use",
                "required": False,
                "action": "store_true",
                "alias": ["--add-eos-token"],
            },
            {
                "arg": "--block_size",
                "help": "Block size to use",
                "required": False,
                "type": int,
                "default": -1,
                "alias": ["--block-size"],
            },
            {
                "arg": "--use_peft",
                "help": "Use PEFT to use",
                "required": False,
                "action": "store_true",
                "alias": ["--use-peft"],
            },
            {
                "arg": "--lora_r",
                "help": "Lora r to use",
                "required": False,
                "type": int,
                "default": 16,
                "alias": ["--lora-r"],
            },
            {
                "arg": "--lora_alpha",
                "help": "Lora alpha to use",
                "required": False,
                "type": int,
                "default": 32,
                "alias": ["--lora-alpha"],
            },
            {
                "arg": "--lora_dropout",
                "help": "Lora dropout to use",
                "required": False,
                "type": float,
                "default": 0.05,
                "alias": ["--lora-dropout"],
            },
            {
                "arg": "--logging_steps",
                "help": "Logging steps to use",
                "required": False,
                "type": int,
                "default": -1,
                "alias": ["--logging-steps"],
            },
            {
                "arg": "--project_name",
                "help": "Output directory",
                "required": False,
                "type": str,
                "alias": ["--project-name"],
            },
            {
                "arg": "--evaluation_strategy",
                "help": "Evaluation strategy to use",
                "required": False,
                "type": str,
                "default": "epoch",
                "alias": ["--evaluation-strategy"],
            },
            {
                "arg": "--save_total_limit",
                "help": "Save total limit to use",
                "required": False,
                "type": int,
                "default": 1,
                "alias": ["--save-total-limit"],
            },
            {
                "arg": "--save_strategy",
                "help": "Save strategy to use",
                "required": False,
                "type": str,
                "default": "epoch",
                "alias": ["--save-strategy"],
            },
            {
                "arg": "--auto_find_batch_size",
                "help": "Auto find batch size True/False",
                "required": False,
                "action": "store_true",
                "alias": ["--auto-find-batch-size"],
            },
            {
                "arg": "--fp16",
                "help": "FP16 True/False",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--push_to_hub",
                "help": "Push to hub True/False. In case you want to push the trained model to huggingface hub",
                "required": False,
                "action": "store_true",
                "alias": ["--push-to-hub"],
            },
            {
                "arg": "--use_int8",
                "help": "Use int8 True/False",
                "required": False,
                "action": "store_true",
                "alias": ["--use-int8"],
            },
            {
                "arg": "--model_max_length",
                "help": "Model max length to use",
                "required": False,
                "type": int,
                "default": 1024,
                "alias": ["--max-len", "--max-length"],
            },
            {
                "arg": "--repo_id",
                "help": "Repo id for hugging face hub. Format is username/repo_name",
                "required": False,
                "type": str,
                "alias": ["--repo-id"],
            },
            {
                "arg": "--use_int4",
                "help": "Use int4 True/False",
                "required": False,
                "action": "store_true",
                "alias": ["--use-int4"],
            },
            {
                "arg": "--trainer",
                "help": "Trainer type to use",
                "required": False,
                "type": str,
                "default": "default",
            },
            {
                "arg": "--target_modules",
                "help": "Target modules to use",
                "required": False,
                "type": str,
                "default": None,
                "alias": ["--target-modules"],
            },
            {
                "arg": "--merge_adapter",
                "help": "Use this flag to merge PEFT adapter with the model",
                "required": False,
                "action": "store_true",
                "alias": ["--merge-adapter"],
            },
            {
                "arg": "--token",
                "help": "Hugingface token to use",
                "required": False,
                "type": str,
            },
            {
                "arg": "--backend",
                "help": "Backend to use: default or spaces. Spaces backend requires push_to_hub and repo_id",
                "required": False,
                "type": str,
                "default": "default",
            },
            {
                "arg": "--username",
                "help": "Huggingface username to use",
                "required": False,
                "type": str,
            },
            {
                "arg": "--use_flash_attention_2",
                "help": "Use flash attention 2",
                "required": False,
                "action": "store_true",
                "alias": ["--use-flash-attention-2", "--use-fa2"],
            },
            {
                "arg": "--disable_gradient_checkpointing",
                "help": "Disable gradient checkpointing",
                "required": False,
                "action": "store_true",
                "alias": ["--disable-gradient-checkpointing", "--disable-gc"],
            },
        ]
        run_llm_parser = parser.add_parser("llm", description="✨ Run AutoTrain LLM")
        for arg in arg_list:
            names = [arg["arg"]] + arg.get("alias", [])
            if "action" in arg:
                run_llm_parser.add_argument(
                    *names,
                    dest=arg["arg"].replace("--", "").replace("-", "_"),
                    help=arg["help"],
                    required=arg.get("required", False),
                    action=arg.get("action"),
                    default=arg.get("default"),
                )
            else:
                run_llm_parser.add_argument(
                    *names,
                    dest=arg["arg"].replace("--", "").replace("-", "_"),
                    help=arg["help"],
                    required=arg.get("required", False),
                    type=arg.get("type"),
                    default=arg.get("default"),
                )
        run_llm_parser.set_defaults(func=run_llm_command_factory)

    def __init__(self, args):
        self.args = args

        store_true_arg_names = [
            "train",
            "deploy",
            "inference",
            "add_eos_token",
            "use_peft",
            "auto_find_batch_size",
            "fp16",
            "push_to_hub",
            "use_int8",
            "use_int4",
            "merge_adapter",
            "use_flash_attention_2",
            "disable_gradient_checkpointing",
        ]
        for arg_name in store_true_arg_names:
            if getattr(self.args, arg_name) is None:
                setattr(self.args, arg_name, False)

        if self.args.train:
            if self.args.project_name is None:
                raise ValueError("Project name must be specified")
            if self.args.data_path is None:
                raise ValueError("Data path must be specified")
            if self.args.model is None:
                raise ValueError("Model must be specified")
            if self.args.push_to_hub:
                # must have project_name, username and token OR project_name, repo_id, token
                if self.args.username is None and self.args.repo_id is None:
                    raise ValueError("Username or repo id must be specified for push to hub")
                if self.args.token is None:
                    raise ValueError("Token must be specified for push to hub")

            if self.args.backend.startswith("spaces") or self.args.backend.startswith("ep-"):
                if not self.args.push_to_hub:
                    raise ValueError("Push to hub must be specified for spaces backend")
                if self.args.repo_id is None:
                    raise ValueError("Repo id must be specified for spaces backend")
                if self.args.token is None:
                    raise ValueError("Token must be specified for spaces backend")

        if self.args.inference:
            from autotrain.infer.text_generation import TextGenerationInference

            tgi = TextGenerationInference(
                self.args.project_name,
                use_int4=self.args.use_int4,
                use_int8=self.args.use_int8,
            )
            while True:
                prompt = input("User: ")
                if prompt == "exit()":
                    break
                print(f"Bot: {tgi.chat(prompt)}")

        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()

        if not cuda_available and not mps_available:
            raise ValueError("No GPU/MPS device found. LLM training requires an accelerator")

        if cuda_available:
            self.num_gpus = torch.cuda.device_count()
        elif mps_available:
            self.num_gpus = 1

    def run(self):
        from autotrain.backend import EndpointsRunner, SpaceRunner
        from autotrain.trainers.clm.__main__ import train as train_llm
        from autotrain.trainers.clm.params import LLMTrainingParams

        logger.info("Running LLM")
        logger.info(f"Params: {self.args}")
        if self.args.train:
            params = LLMTrainingParams(
                model=self.args.model,
                data_path=self.args.data_path,
                train_split=self.args.train_split,
                valid_split=self.args.valid_split,
                text_column=self.args.text_column,
                lr=self.args.learning_rate,
                epochs=self.args.num_train_epochs,
                batch_size=self.args.train_batch_size,
                warmup_ratio=self.args.warmup_ratio,
                gradient_accumulation=self.args.gradient_accumulation_steps,
                optimizer=self.args.optimizer,
                scheduler=self.args.scheduler,
                weight_decay=self.args.weight_decay,
                max_grad_norm=self.args.max_grad_norm,
                seed=self.args.seed,
                add_eos_token=self.args.add_eos_token,
                block_size=self.args.block_size,
                use_peft=self.args.use_peft,
                lora_r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                logging_steps=self.args.logging_steps,
                project_name=self.args.project_name,
                evaluation_strategy=self.args.evaluation_strategy,
                save_total_limit=self.args.save_total_limit,
                save_strategy=self.args.save_strategy,
                auto_find_batch_size=self.args.auto_find_batch_size,
                fp16=self.args.fp16,
                push_to_hub=self.args.push_to_hub,
                use_int8=self.args.use_int8,
                model_max_length=self.args.model_max_length,
                repo_id=self.args.repo_id,
                use_int4=self.args.use_int4,
                trainer=self.args.trainer,
                target_modules=self.args.target_modules,
                token=self.args.token,
                merge_adapter=self.args.merge_adapter,
                username=self.args.username,
                use_flash_attention_2=self.args.use_flash_attention_2,
                rejected_text_column=self.args.rejected_text_column,
                disable_gradient_checkpointing=self.args.disable_gradient_checkpointing,
            )

            # space training
            if self.args.backend.startswith("spaces"):
                logger.info("Creating space...")
                sr = SpaceRunner(
                    params=params,
                    backend=self.args.backend,
                )
                space_id = sr.prepare()
                logger.info(f"Training Space created. Check progress at https://hf.co/spaces/{space_id}")
                sys.exit(0)

            if self.args.backend.startswith("ep-"):
                logger.info("Creating training endpoint...")
                sr = EndpointsRunner(
                    params=params,
                    backend=self.args.backend,
                )
                sr.prepare()
                logger.info("Training endpoint created.")
                sys.exit(0)

            # local training
            params.save(output_dir=self.args.project_name)
            if self.num_gpus == 1:
                train_llm(params)
            else:
                cmd = [
                    "accelerate",
                    "launch",
                    "--multi_gpu",
                    "--num_machines",
                    "1",
                    "--num_processes",
                ]
                cmd.append(str(self.num_gpus))
                cmd.append("--mixed_precision")
                if self.args.fp16:
                    cmd.append("fp16")
                else:
                    cmd.append("no")

                cmd.extend(
                    [
                        "-m",
                        "autotrain.trainers.clm",
                        "--training_config",
                        os.path.join(self.args.project_name, "training_params.json"),
                    ]
                )

                env = os.environ.copy()
                process = subprocess.Popen(cmd, env=env)
                process.wait()
