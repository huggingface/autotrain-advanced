import os
import subprocess
from argparse import ArgumentParser

import torch
from loguru import logger

from autotrain.infer.text_generation import TextGenerationInference

from ..trainers.clm import train as train_llm
from ..trainers.utils import LLMTrainingParams
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
            },
            {
                "arg": "--train_split",
                "help": "Test dataset split to use",
                "required": False,
                "type": str,
                "default": "train",
            },
            {
                "arg": "--valid_split",
                "help": "Validation dataset split to use",
                "required": False,
                "type": str,
                "default": None,
            },
            {
                "arg": "--text_column",
                "help": "Text column to use",
                "required": False,
                "type": str,
                "default": "text",
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
            },
            {
                "arg": "--num_train_epochs",
                "help": "Number of training epochs to use",
                "required": False,
                "type": int,
                "default": 1,
            },
            {
                "arg": "--train_batch_size",
                "help": "Training batch size to use",
                "required": False,
                "type": int,
                "default": 2,
            },
            {
                "arg": "--eval_batch_size",
                "help": "Evaluation batch size to use",
                "required": False,
                "type": int,
                "default": 4,
            },
            {
                "arg": "--warmup_ratio",
                "help": "Warmup proportion to use",
                "required": False,
                "type": float,
                "default": 0.1,
            },
            {
                "arg": "--gradient_accumulation_steps",
                "help": "Gradient accumulation steps to use",
                "required": False,
                "type": int,
                "default": 1,
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
            },
            {
                "arg": "--max_grad_norm",
                "help": "Max gradient norm to use",
                "required": False,
                "type": float,
                "default": 1.0,
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
            },
            {
                "arg": "--block_size",
                "help": "Block size to use",
                "required": False,
                "type": int,
                "default": -1,
            },
            {
                "arg": "--use_peft",
                "help": "Use PEFT to use",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--lora_r",
                "help": "Lora r to use",
                "required": False,
                "type": int,
                "default": 16,
            },
            {
                "arg": "--lora_alpha",
                "help": "Lora alpha to use",
                "required": False,
                "type": int,
                "default": 32,
            },
            {
                "arg": "--lora_dropout",
                "help": "Lora dropout to use",
                "required": False,
                "type": float,
                "default": 0.05,
            },
            {
                "arg": "--training_type",
                "help": "Training type to use",
                "required": False,
                "type": str,
                "default": "generic",
            },
            {
                "arg": "--train_on_inputs",
                "help": "Train on inputs to use",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--logging_steps",
                "help": "Logging steps to use",
                "required": False,
                "type": int,
                "default": -1,
            },
            {
                "arg": "--project_name",
                "help": "Output directory",
                "required": False,
                "type": str,
            },
            {
                "arg": "--evaluation_strategy",
                "help": "Evaluation strategy to use",
                "required": False,
                "type": str,
                "default": "epoch",
            },
            {
                "arg": "--save_total_limit",
                "help": "Save total limit to use",
                "required": False,
                "type": int,
                "default": 1,
            },
            {
                "arg": "--save_strategy",
                "help": "Save strategy to use",
                "required": False,
                "type": str,
                "default": "epoch",
            },
            {
                "arg": "--auto_find_batch_size",
                "help": "Auto find batch size True/False",
                "required": False,
                "action": "store_true",
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
            },
            {
                "arg": "--use_int8",
                "help": "Use int8 True/False",
                "required": False,
                "action": "store_true",
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
                "help": "Repo id for hugging face hub",
                "required": False,
                "type": str,
            },
            {
                "arg": "--use_int4",
                "help": "Use int4 True/False",
                "required": False,
                "action": "store_true",
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
            },
        ]
        run_llm_parser = parser.add_parser("llm", description="✨ Run AutoTrain LLM")
        for arg in arg_list:
            if "action" in arg:
                run_llm_parser.add_argument(
                    arg["arg"],
                    help=arg["help"],
                    required=arg.get("required", False),
                    action=arg.get("action"),
                    default=arg.get("default"),
                )
            else:
                run_llm_parser.add_argument(
                    arg["arg"],
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
            "train_on_inputs",
            "auto_find_batch_size",
            "fp16",
            "push_to_hub",
            "use_int8",
            "use_int4",
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
                if self.args.repo_id is None:
                    raise ValueError("Repo id must be specified for push to hub")

        if self.args.inference:
            tgi = TextGenerationInference(
                self.args.project_name or self.args.model, use_int4=self.args.use_int4, use_int8=self.args.use_int8
            )
            while True:
                prompt = input("User: ")
                if prompt == "exit()":
                    break
                print(f"Bot: {tgi.chat(prompt)}")

        if not torch.cuda.is_available():
            raise ValueError("No GPU found. Please install CUDA and try again.")

        self.num_gpus = torch.cuda.device_count()

    def run(self):
        logger.info("Running LLM")
        logger.info(f"Params: {self.args}")
        if self.args.train:
            params = LLMTrainingParams(
                model_name=self.args.model,
                data_path=self.args.data_path,
                train_split=self.args.train_split,
                valid_split=self.args.valid_split,
                text_column=self.args.text_column,
                learning_rate=self.args.learning_rate,
                num_train_epochs=self.args.num_train_epochs,
                train_batch_size=self.args.train_batch_size,
                eval_batch_size=self.args.eval_batch_size,
                warmup_ratio=self.args.warmup_ratio,
                gradient_accumulation_steps=self.args.gradient_accumulation_steps,
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
                training_type=self.args.training_type,
                train_on_inputs=self.args.train_on_inputs,
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
            )
            params.save(output_dir=self.args.project_name)
            if self.num_gpus == 1:
                train_llm(params)
            else:
                cmd = ["accelerate", "launch", "--multi_gpu", "--num_machines", "1", "--num_processes"]
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
