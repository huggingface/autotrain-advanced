from argparse import ArgumentParser

from autotrain import logger
from autotrain.cli.utils import common_args, llm_munge_data
from autotrain.project import AutoTrainProject
from autotrain.trainers.clm.params import LLMTrainingParams

from . import BaseAutoTrainCommand


def run_llm_command_factory(args):
    return RunAutoTrainLLMCommand(args)


class RunAutoTrainLLMCommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        arg_list = [
            {
                "arg": "--text_column",
                "help": "Specify the dataset column to use for text data. This parameter is essential for models processing textual information. Default is 'text'.",
                "required": False,
                "type": str,
                "default": "text",
                "alias": ["--text-column"],
            },
            {
                "arg": "--rejected_text_column",
                "help": "Define the column to use for storing rejected text entries, which are typically entries that do not meet certain criteria for processing. Default is 'rejected'. Used only for orpo, dpo and reward trainerss",
                "required": False,
                "type": str,
                "default": "rejected",
                "alias": ["--rejected-text-column"],
            },
            {
                "arg": "--prompt-text-column",
                "help": "Identify the column that contains prompt text for tasks requiring contextual inputs, such as conversation or completion generation. Default is 'prompt'. Used only for dpo trainer",
                "required": False,
                "type": str,
                "default": "prompt",
                "alias": ["--prompt-text-column"],
            },
            {
                "arg": "--model-ref",
                "help": "Reference model to use for DPO when not using PEFT",
                "required": False,
                "type": str,
            },
            {
                "arg": "--warmup_ratio",
                "help": "Set the proportion of training allocated to warming up the learning rate, which can enhance model stability and performance at the start of training. Default is 0.1",
                "required": False,
                "type": float,
                "default": 0.1,
                "alias": ["--warmup-ratio"],
            },
            {
                "arg": "--optimizer",
                "help": "Choose the optimizer algorithm for training the model. Different optimizers can affect the training speed and model performance. 'adamw_torch' is used by default.",
                "required": False,
                "type": str,
                "default": "adamw_torch",
            },
            {
                "arg": "--scheduler",
                "help": "Select the learning rate scheduler to adjust the learning rate based on the number of epochs. 'linear' decreases the learning rate linearly from the initial lr set. Default is 'linear'. Try 'cosine' for a cosine annealing schedule.",
                "required": False,
                "type": str,
                "default": "linear",
            },
            {
                "arg": "--weight_decay",
                "help": "Define the weight decay rate for regularization, which helps prevent overfitting by penalizing larger weights. Default is 0.0",
                "required": False,
                "type": float,
                "default": 0.0,
                "alias": ["--weight-decay"],
            },
            {
                "arg": "--max_grad_norm",
                "help": "Set the maximum norm for gradient clipping, which is critical for preventing gradients from exploding during backpropagation. Default is 1.0.",
                "required": False,
                "type": float,
                "default": 1.0,
                "alias": ["--max-grad-norm"],
            },
            {
                "arg": "--add_eos_token",
                "help": "Toggle whether to automatically add an End Of Sentence (EOS) token at the end of texts, which can be critical for certain types of models like language models. Only used for `default` trainer",
                "required": False,
                "action": "store_true",
                "alias": ["--add-eos-token"],
            },
            {
                "arg": "--block_size",
                "help": "Specify the block size for processing sequences. This is maximum sequence length or length of one block of text. Setting to -1 determines block size automatically. Default is -1.",
                "required": False,
                "type": str,
                "default": "-1",
                "alias": ["--block-size"],
            },
            {
                "arg": "--peft",
                "help": "Enable LoRA-PEFT",
                "required": False,
                "action": "store_true",
                "alias": ["--use-peft"],
            },
            {
                "arg": "--lora_r",
                "help": "Set the 'r' parameter for Low-Rank Adaptation (LoRA). Default is 16.",
                "required": False,
                "type": int,
                "default": 16,
                "alias": ["--lora-r"],
            },
            {
                "arg": "--lora_alpha",
                "help": "Specify the 'alpha' parameter for LoRA. Default is 32.",
                "required": False,
                "type": int,
                "default": 32,
                "alias": ["--lora-alpha"],
            },
            {
                "arg": "--lora_dropout",
                "help": "Set the dropout rate within the LoRA layers to help prevent overfitting during adaptation. Default is 0.05.",
                "required": False,
                "type": float,
                "default": 0.05,
                "alias": ["--lora-dropout"],
            },
            {
                "arg": "--logging_steps",
                "help": "Determine how often to log training progress in terms of steps. Setting it to '-1' determines logging steps automatically.",
                "required": False,
                "type": int,
                "default": -1,
                "alias": ["--logging-steps"],
            },
            {
                "arg": "--evaluation_strategy",
                "help": "Choose how frequently to evaluate the model's performance, with 'epoch' as the default, meaning at the end of each training epoch",
                "required": False,
                "type": str,
                "default": "epoch",
                "alias": ["--evaluation-strategy"],
                "choices": ["epoch", "steps", "no"],
            },
            {
                "arg": "--save_total_limit",
                "help": "Limit the total number of saved model checkpoints to manage disk usage effectively. Default is to save only the latest checkpoint",
                "required": False,
                "type": int,
                "default": 1,
                "alias": ["--save-total-limit"],
            },
            {
                "arg": "--auto_find_batch_size",
                "help": "Automatically determine the optimal batch size based on system capabilities to maximize efficiency.",
                "required": False,
                "action": "store_true",
                "alias": ["--auto-find-batch-size"],
            },
            {
                "arg": "--mixed_precision",
                "help": "Choose the precision mode for training to optimize performance and memory usage. Options are 'fp16', 'bf16', or None for default precision. Default is None.",
                "required": False,
                "type": str,
                "default": None,
                "choices": ["fp16", "bf16", None],
                "alias": ["--mixed-precision"],
            },
            {
                "arg": "--quantization",
                "help": "Choose the quantization level to reduce model size and potentially increase inference speed. Options include 'int4', 'int8', or None. Enabling requires --peft",
                "required": False,
                "type": str,
                "default": None,
                "alias": ["--quantization"],
                "choices": ["int4", "int8", None],
            },
            {
                "arg": "--model_max_length",
                "help": "Set the maximum length for the model to process in a single batch, which can affect both performance and memory usage. Default is 1024",
                "required": False,
                "type": int,
                "default": 1024,
                "alias": ["--model-max-length"],
            },
            {
                "arg": "--max_prompt_length",
                "help": "Specify the maximum length for prompts used in training, particularly relevant for tasks requiring initial contextual input. Used only for `orpo` trainer.",
                "required": False,
                "type": int,
                "default": 128,
                "alias": ["--max-prompt-length"],
            },
            {
                "arg": "--max_completion_length",
                "help": "Completion length to use, for orpo: encoder-decoder models only",
                "required": False,
                "type": int,
                "default": None,
                "alias": ["--max-completion-length"],
            },
            {
                "arg": "--trainer",
                "help": "Trainer type to use",
                "required": False,
                "type": str,
                "default": "default",
                "choices": ["default", "dpo", "sft", "orpo", "reward"],
            },
            {
                "arg": "--target_modules",
                "help": "Identify specific modules within the model architecture to target with adaptations or optimizations, such as LoRA. Comma separated list of module names. Default is 'all-linear'.",
                "required": False,
                "type": str,
                "default": "all-linear",
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
                "arg": "--use_flash_attention_2",
                "help": "Use flash attention 2",
                "required": False,
                "action": "store_true",
                "alias": ["--use-flash-attention-2", "--use-fa2"],
            },
            {
                "arg": "--dpo-beta",
                "help": "Beta for DPO trainer",
                "required": False,
                "type": float,
                "default": 0.1,
                "alias": ["--dpo-beta"],
            },
            {
                "arg": "--chat_template",
                "help": "Apply a specific template for chat-based interactions, with options including 'tokenizer', 'chatml', 'zephyr', or None. This setting can shape the model's conversational behavior. ",
                "required": False,
                "default": None,
                "alias": ["--chat-template"],
                "choices": ["tokenizer", "chatml", "zephyr", None],
            },
            {
                "arg": "--padding",
                "help": "Specify the padding direction for sequences, critical for models sensitive to input alignment. Options include 'left', 'right', or None",
                "required": False,
                "type": str,
                "default": None,
                "alias": ["--padding"],
                "choices": ["left", "right", None],
            },
        ]
        arg_list = common_args() + arg_list
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
                    choices=arg.get("choices"),
                )
        run_llm_parser.set_defaults(func=run_llm_command_factory)

    def __init__(self, args):
        self.args = args

        store_true_arg_names = [
            "train",
            "deploy",
            "inference",
            "add_eos_token",
            "peft",
            "auto_find_batch_size",
            "push_to_hub",
            "merge_adapter",
            "use_flash_attention_2",
            "disable_gradient_checkpointing",
        ]
        for arg_name in store_true_arg_names:
            if getattr(self.args, arg_name) is None:
                setattr(self.args, arg_name, False)

        block_size_split = self.args.block_size.strip().split(",")
        if len(block_size_split) == 1:
            self.args.block_size = int(block_size_split[0])
        elif len(block_size_split) > 1:
            self.args.block_size = [int(x.strip()) for x in block_size_split]
        else:
            raise ValueError("Invalid block size")

        if self.args.train:
            if self.args.project_name is None:
                raise ValueError("Project name must be specified")
            if self.args.data_path is None:
                raise ValueError("Data path must be specified")
            if self.args.model is None:
                raise ValueError("Model must be specified")
            if self.args.push_to_hub:
                # must have project_name, username and token OR project_name, token
                if self.args.username is None:
                    raise ValueError("Usernamemust be specified for push to hub")
                if self.args.token is None:
                    raise ValueError("Token must be specified for push to hub")

            if self.args.backend.startswith("spaces") or self.args.backend.startswith("ep-"):
                if not self.args.push_to_hub:
                    raise ValueError("Push to hub must be specified for spaces backend")
                if self.args.username is None:
                    raise ValueError("Username must be specified for spaces backend")
                if self.args.token is None:
                    raise ValueError("Token must be specified for spaces backend")

        if self.args.deploy:
            raise NotImplementedError("Deploy is not implemented yet")
        if self.args.inference:
            raise NotImplementedError("Inference is not implemented yet")

    def run(self):
        logger.info("Running LLM")
        if self.args.train:
            params = LLMTrainingParams(**vars(self.args))
            params = llm_munge_data(params, local=self.args.backend.startswith("local"))
            project = AutoTrainProject(params=params, backend=self.args.backend)
            job_id = project.create()
            logger.info(f"Job ID: {job_id}")
