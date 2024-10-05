from argparse import ArgumentParser

from autotrain import logger
from autotrain.cli.utils import get_field_info
from autotrain.datagen.gen import AutoTrainGen
from autotrain.datagen.params import AutoTrainGenParams

from . import BaseAutoTrainCommand


def run_autotrain_gen_command(args):
    return RunAutoTrainGenCommand(args)


class RunAutoTrainGenCommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        arg_list = get_field_info(AutoTrainGenParams)
        run_autotrain_gen_parser = parser.add_parser("gen", description="✨ AutoTrain Gen")
        for arg in arg_list:
            names = [arg["arg"]] + arg.get("alias", [])
            if "action" in arg:
                run_autotrain_gen_parser.add_argument(
                    *names,
                    dest=arg["arg"].replace("--", "").replace("-", "_"),
                    help=arg["help"],
                    required=arg.get("required", False),
                    action=arg.get("action"),
                    default=arg.get("default"),
                )
            else:
                run_autotrain_gen_parser.add_argument(
                    *names,
                    dest=arg["arg"].replace("--", "").replace("-", "_"),
                    help=arg["help"],
                    required=arg.get("required", False),
                    type=arg.get("type"),
                    default=arg.get("default"),
                    choices=arg.get("choices"),
                )
        run_autotrain_gen_parser.set_defaults(func=run_autotrain_gen_command)

    def __init__(self, args):
        self.args = args

        store_true_arg_names = [
            "push_to_hub",
        ]
        for arg_name in store_true_arg_names:
            if getattr(self.args, arg_name) is None:
                setattr(self.args, arg_name, False)

    def run(self):
        logger.info("Running AutoTrain Gen 🚀")
        params = AutoTrainGenParams(**vars(self.args))
        AutoTrainGen(params).run()
