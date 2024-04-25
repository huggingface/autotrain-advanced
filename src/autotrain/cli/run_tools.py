from argparse import ArgumentParser

from . import BaseAutoTrainCommand


def run_tools_command_factory(args):
    return RunAutoTrainToolsCommand(args)


class RunAutoTrainToolsCommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        run_app_parser = parser.add_parser("tools", help="Run AutoTrain tools")
        subparsers = run_app_parser.add_subparsers(title="tools", dest="tool_name")

        merge_llm_parser = subparsers.add_parser(
            "merge-llm-adapter",
            help="Merge LLM Adapter tool",
        )
        merge_llm_parser.add_argument(
            "--base-model-path",
            type=str,
            help="Base model path",
        )
        merge_llm_parser.add_argument(
            "--adapter-path",
            type=str,
            help="Adapter path",
        )
        merge_llm_parser.add_argument(
            "--token",
            type=str,
            help="Token",
            default=None,
            required=False,
        )
        merge_llm_parser.add_argument(
            "--pad-to-multiple-of",
            type=int,
            help="Pad to multiple of",
            default=None,
            required=False,
        )
        merge_llm_parser.add_argument(
            "--output-folder",
            type=str,
            help="Output folder",
            required=False,
            default=None,
        )
        merge_llm_parser.add_argument(
            "--push-to-hub",
            action="store_true",
            help="Push to Hugging Face Hub",
            required=False,
        )
        merge_llm_parser.set_defaults(func=run_tools_command_factory, merge_llm_adapter=True)

        convert_to_kohya_parser = subparsers.add_parser("convert_to_kohya", help="Convert to Kohya tool")
        convert_to_kohya_parser.add_argument(
            "--input-path",
            type=str,
            help="Input path",
        )
        convert_to_kohya_parser.add_argument(
            "--output-path",
            type=str,
            help="Output path",
        )
        convert_to_kohya_parser.set_defaults(func=run_tools_command_factory, convert_to_kohya=True)

    def __init__(self, args):
        self.args = args

    def run(self):
        if getattr(self.args, "merge_llm_adapter", False):
            self.run_merge_llm_adapter()
        if getattr(self.args, "convert_to_kohya", False):
            self.run_convert_to_kohya()

    def run_merge_llm_adapter(self):
        from autotrain.tools.merge_adapter import merge_llm_adapter

        merge_llm_adapter(
            base_model_path=self.args.base_model_path,
            adapter_path=self.args.adapter_path,
            token=self.args.token,
            output_folder=self.args.output_folder,
            pad_to_multiple_of=self.args.pad_to_multiple_of,
            push_to_hub=self.args.push_to_hub,
        )

    def run_convert_to_kohya(self):
        from autotrain.tools.convert_to_kohya import convert_to_kohya

        convert_to_kohya(
            input_path=self.args.input_path,
            output_path=self.args.output_path,
        )
