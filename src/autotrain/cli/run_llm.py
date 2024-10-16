from argparse import ArgumentParser

from autotrain import logger
from autotrain.cli.utils import get_field_info
from autotrain.project import AutoTrainProject
from autotrain.trainers.clm.params import LLMTrainingParams

from . import BaseAutoTrainCommand


def run_llm_command_factory(args):
    return RunAutoTrainLLMCommand(args)


class RunAutoTrainLLMCommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        arg_list = get_field_info(LLMTrainingParams)
        arg_list = [
            {
                "arg": "--train",
                "help": "Command to train the model",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--deploy",
                "help": "Command to deploy the model (limited availability)",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--inference",
                "help": "Command to run inference (limited availability)",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--backend",
                "help": "Backend",
                "required": False,
                "type": str,
                "default": "local",
            },
        ] + arg_list
        arg_list = [arg for arg in arg_list if arg["arg"] != "--block-size"]
        arg_list.append(
            {
                "arg": "--block_size",
                "help": "Block size",
                "required": False,
                "type": str,
                "default": "1024",
                "alias": ["--block-size"],
            }
        )
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
            project = AutoTrainProject(params=params, backend=self.args.backend, process=True)
            job_id = project.create()
            logger.info(f"Job ID: {job_id}")
