from argparse import ArgumentParser

from autotrain.backend import SpaceRunner
from autotrain.trainers.generic.params import GenericParams
from autotrain.trainers.generic.utils import create_dataset_repo

from . import BaseAutoTrainCommand


BACKEND_CHOICES = [
    "spaces-a10gl",
    "spaces-a10gs",
    "spaces-a100",
    "spaces-t4m",
    "spaces-t4s",
    "spaces-cpu",
    "spaces-cpuf",
]


def run_spacerunner_command_factory(args):
    return RunAutoTrainSpaceRunnerCommand(args)


class RunAutoTrainSpaceRunnerCommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        arg_list = [
            {
                "arg": "--project-name",
                "help": "Name of the project. Must be unique.",
                "required": True,
                "type": str,
            },
            {
                "arg": "--script-path",
                "help": "Path to the script",
                "required": True,
                "type": str,
            },
            {
                "arg": "--username",
                "help": "Hugging Face Username, can also be an organization name",
                "required": True,
                "type": str,
            },
            {
                "arg": "--token",
                "help": "Hugging Face API Token",
                "required": True,
                "type": str,
            },
            {
                "arg": "--backend",
                "help": "Hugging Face backend to use",
                "required": True,
                "type": str,
                "choices": BACKEND_CHOICES,
            },
            {
                "arg": "--env",
                "help": "Environment variables, e.g. --env FOO=bar;FOO2=bar2;FOO3=bar3",
                "required": False,
                "type": str,
            },
            {
                "arg": "--args",
                "help": "Arguments to pass to the script, e.g. --args foo=bar;foo2=bar2;foo3=bar3;store_true_arg",
                "required": False,
                "type": str,
            },
        ]
        run_spacerunner_parser = parser.add_parser("spacerunner", description="✨ Run AutoTrain SpaceRunner")
        for arg in arg_list:
            names = [arg["arg"]] + arg.get("alias", [])
            if "action" in arg:
                run_spacerunner_parser.add_argument(
                    *names,
                    dest=arg["arg"].replace("--", "").replace("-", "_"),
                    help=arg["help"],
                    required=arg.get("required", False),
                    action=arg.get("action"),
                    default=arg.get("default"),
                    choices=arg.get("choices"),
                )
            else:
                run_spacerunner_parser.add_argument(
                    *names,
                    dest=arg["arg"].replace("--", "").replace("-", "_"),
                    help=arg["help"],
                    required=arg.get("required", False),
                    type=arg.get("type"),
                    default=arg.get("default"),
                    choices=arg.get("choices"),
                )
        run_spacerunner_parser.set_defaults(func=run_spacerunner_command_factory)

    def __init__(self, args):
        self.args = args

        store_true_arg_names = []
        for arg_name in store_true_arg_names:
            if getattr(self.args, arg_name) is None:
                setattr(self.args, arg_name, False)

        env_vars = {}
        if self.args.env:
            for env_name_value in self.args.env.split(";"):
                if len(env_name_value.split("=")) == 2:
                    env_vars[env_name_value.split("=")[0]] = env_name_value.split("=")[1]
                else:
                    raise ValueError("Invalid environment variable format.")
        self.args.env = env_vars

        app_args = {}
        store_true_args = []
        if self.args.args:
            for arg_name_value in self.args.args.split(";"):
                if len(arg_name_value.split("=")) == 1:
                    store_true_args.append(arg_name_value)
                elif len(arg_name_value.split("=")) == 2:
                    app_args[arg_name_value.split("=")[0]] = arg_name_value.split("=")[1]
                else:
                    raise ValueError("Invalid argument format.")

        for arg_name in store_true_args:
            app_args[arg_name] = ""
        self.args.args = app_args

    def run(self):
        dataset_id = create_dataset_repo(
            username=self.args.username,
            project_name=self.args.project_name,
            script_path=self.args.script_path,
            token=self.args.token,
        )
        params = GenericParams(
            project_name=self.args.project_name,
            data_path=dataset_id,
            username=self.args.username,
            token=self.args.token,
            backend=self.args.backend,
            script_path=self.args.script_path,
            env=self.args.env,
            args=self.args.args,
            repo_id=f"{self.args.username}/{self.args.project_name}",
        )
        sr = SpaceRunner(params=params, backend=self.args.backend)
        sr.prepare()
