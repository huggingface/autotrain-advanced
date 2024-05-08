from argparse import ArgumentParser

from autotrain import logger
from autotrain.cli.utils import common_args, seq2seq_munge_data
from autotrain.project import AutoTrainProject
from autotrain.trainers.seq2seq.params import Seq2SeqParams

from . import BaseAutoTrainCommand


def run_seq2seq_command_factory(args):
    return RunAutoTrainSeq2SeqCommand(args)


class RunAutoTrainSeq2SeqCommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        arg_list = [
            {
                "arg": "--text-column",
                "help": "Specify the column name in the dataset that contains the text data. Useful for distinguishing between multiple text fields. Default is 'text'.",
                "required": False,
                "type": str,
                "default": "text",
            },
            {
                "arg": "--target-column",
                "help": "Specify the column name that holds the target data for training. Helps in distinguishing different potential outputs. Default is 'target'.",
                "required": False,
                "type": str,
                "default": "target",
            },
            {
                "arg": "--max-seq-length",
                "help": "Set the maximum sequence length (number of tokens) that the model should handle in a single input. Longer sequences are truncated. Affects both memory usage and computational requirements. Default is 128 tokens.",
                "required": False,
                "type": int,
                "default": 128,
            },
            {
                "arg": "--max-target-length",
                "help": "Define the maximum number of tokens for the target sequence in each input. Useful for models that generate outputs, ensuring uniformity in sequence length. Default is set to 128 tokens.",
                "required": False,
                "type": int,
                "default": 128,
            },
            {
                "arg": "--warmup-ratio",
                "help": "Define the proportion of training to be dedicated to a linear warmup where learning rate gradually increases. This can help in stabilizing the training process early on. Default ratio is 0.1.",
                "required": False,
                "type": float,
                "default": 0.1,
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
                "arg": "--weight-decay",
                "help": "Set the weight decay rate to apply for regularization. Helps in preventing the model from overfitting by penalizing large weights. Default is 0.0, meaning no weight decay is applied.",
                "required": False,
                "type": float,
                "default": 0.0,
            },
            {
                "arg": "--max-grad-norm",
                "help": "Specify the maximum norm of the gradients for gradient clipping. Gradient clipping is used to prevent the exploding gradient problem in deep neural networks. Default is 1.0.",
                "required": False,
                "type": float,
                "default": 1.0,
            },
            {
                "arg": "--logging-steps",
                "help": "Determine how often to log training progress. Set this to the number of steps between each log output. -1 determines logging steps automatically. Default is -1.",
                "required": False,
                "type": int,
                "default": -1,
            },
            {
                "arg": "--evaluation-strategy",
                "help": "Specify how often to evaluate the model performance. Options include 'no', 'steps', 'epoch'. 'epoch' evaluates at the end of each training epoch by default.",
                "required": False,
                "type": str,
                "default": "epoch",
            },
            {
                "arg": "--save-total-limit",
                "help": "Limit the total number of model checkpoints to save. Helps manage disk space by retaining only the most recent checkpoints. Default is to save only the latest one.",
                "required": False,
                "type": int,
                "default": 1,
            },
            {
                "arg": "--auto-find-batch-size",
                "help": "Enable automatic batch size determination based on your hardware capabilities. When set, it tries to find the largest batch size that fits in memory.",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--mixed-precision",
                "help": "Choose the precision mode for training to optimize performance and memory usage. Options are 'fp16', 'bf16', or None for default precision. Default is None.",
                "required": False,
                "type": str,
                "default": None,
                "choices": ["fp16", "bf16", None],
            },
            {
                "arg": "--peft",
                "help": "Enable LoRA-PEFT",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--quantization",
                "help": "Select the quantization mode to reduce model size and potentially increase inference speed. Options include 'int8' for 8-bit integer quantization or None for no quantization. Default is None",
                "required": False,
                "type": str,
                "default": None,
                "choices": ["int8", None],
            },
            {
                "arg": "--lora-r",
                "help": "Set the rank 'R' for the LoRA (Low-Rank Adaptation) technique. Default is 16.",
                "required": False,
                "type": int,
                "default": 16,
            },
            {
                "arg": "--lora-alpha",
                "help": "Specify the 'Alpha' parameter for LoRA. Default is 32.",
                "required": False,
                "type": int,
                "default": 32,
            },
            {
                "arg": "--lora-dropout",
                "help": "Determine the dropout rate to apply in the LoRA layers, which can help in preventing overfitting by randomly disabling a fraction of neurons during training. Default rate is 0.05.",
                "required": False,
                "type": float,
                "default": 0.05,
            },
            {
                "arg": "--target-modules",
                "help": "List the modules within the model architecture that should be targeted for specific techniques such as LoRA adaptations. Useful for fine-tuning particular components of large models. By default all linear layers are targeted.",
                "required": False,
                "type": str,
                "default": "all-linear",
            },
        ]
        arg_list = common_args() + arg_list
        run_seq2seq_parser = parser.add_parser("seq2seq", description="✨ Run AutoTrain Seq2Seq")
        for arg in arg_list:
            if "action" in arg:
                run_seq2seq_parser.add_argument(
                    arg["arg"],
                    help=arg["help"],
                    required=arg.get("required", False),
                    action=arg.get("action"),
                    default=arg.get("default"),
                )
            else:
                run_seq2seq_parser.add_argument(
                    arg["arg"],
                    help=arg["help"],
                    required=arg.get("required", False),
                    type=arg.get("type"),
                    default=arg.get("default"),
                    choices=arg.get("choices"),
                )
        run_seq2seq_parser.set_defaults(func=run_seq2seq_command_factory)

    def __init__(self, args):
        self.args = args

        store_true_arg_names = ["train", "deploy", "inference", "auto_find_batch_size", "push_to_hub", "peft"]
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
                if self.args.username is None:
                    raise ValueError("Username must be specified for push to hub")
        else:
            raise ValueError("Must specify --train, --deploy or --inference")

    def run(self):
        logger.info("Running Seq2Seq Classification")
        if self.args.train:
            params = Seq2SeqParams(**vars(self.args))
            params = seq2seq_munge_data(params, local=self.args.backend.startswith("local"))
            project = AutoTrainProject(params=params, backend=self.args.backend)
            job_id = project.create()
            logger.info(f"Job ID: {job_id}")
