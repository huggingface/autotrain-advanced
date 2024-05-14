from argparse import ArgumentParser

from autotrain import logger
from autotrain.cli.utils import common_args, token_clf_munge_data
from autotrain.project import AutoTrainProject
from autotrain.trainers.token_classification.params import TokenClassificationParams

from . import BaseAutoTrainCommand


def run_token_classification_command_factory(args):
    return RunAutoTrainTokenClassificationCommand(args)


class RunAutoTrainTokenClassificationCommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        arg_list = [
            {
                "arg": "--tokens-column",
                "help": "Tokens column to use. Must be a stringified list of tokens if using a CSV file. Default is 'tokens'.",
                "required": False,
                "type": str,
                "default": "tokens",
            },
            {
                "arg": "--tags-column",
                "help": "Tags column to use. Must be a stringified list of tags if using a CSV file. Default is 'tags'.",
                "required": False,
                "type": str,
                "default": "tags",
            },
            {
                "arg": "--max-seq-length",
                "help": "Set the maximum sequence length (number of tokens) that the model should handle in a single input. Longer sequences are truncated. Affects both memory usage and computational requirements. Default is 128 tokens.",
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
                "choices": ["steps", "epoch", "no"],
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
                "arg": "--early-stopping-patience",
                "help": "Specify the number of epochs with no improvement after which training will stop. Default is 5.",
                "required": False,
                "type": int,
                "default": 5,
            },
            {
                "arg": "--early-stopping-threshold",
                "help": "Define the minimum change in the monitored metric to qualify as an improvement. Default is 0.01.",
                "required": False,
                "type": float,
                "default": 0.01,
            },
        ]
        arg_list = common_args() + arg_list
        arg_list = [arg for arg in arg_list if arg["arg"] != "--disable-gradient-checkpointing"]
        run_token_classification_parser = parser.add_parser(
            "token-classification", description="✨ Run AutoTrain Token Classification"
        )
        for arg in arg_list:
            if "action" in arg:
                run_token_classification_parser.add_argument(
                    arg["arg"],
                    help=arg["help"],
                    required=arg.get("required", False),
                    action=arg.get("action"),
                    default=arg.get("default"),
                )
            else:
                run_token_classification_parser.add_argument(
                    arg["arg"],
                    help=arg["help"],
                    required=arg.get("required", False),
                    type=arg.get("type"),
                    default=arg.get("default"),
                    choices=arg.get("choices"),
                )
        run_token_classification_parser.set_defaults(func=run_token_classification_command_factory)

    def __init__(self, args):
        self.args = args

        store_true_arg_names = [
            "train",
            "deploy",
            "inference",
            "auto_find_batch_size",
            "push_to_hub",
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
                if self.args.username is None:
                    raise ValueError("Username must be specified for push to hub")
        else:
            raise ValueError("Must specify --train, --deploy or --inference")

    def run(self):
        logger.info("Running Token Classification")
        if self.args.train:
            params = TokenClassificationParams(**vars(self.args))
            params = token_clf_munge_data(params, local=self.args.backend.startswith("local"))
            project = AutoTrainProject(params=params, backend=self.args.backend)
            job_id = project.create()
            logger.info(f"Job ID: {job_id}")
