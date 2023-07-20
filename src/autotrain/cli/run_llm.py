from argparse import ArgumentParser

from loguru import logger

from autotrain.infer.text_generation import TextGenerationInference

from ..trainers.clm import train as train_llm
from ..trainers.utils import LLMTrainingParams
from . import BaseAutoTrainCommand


def run_llm_command_factory(args):
    return RunAutoTrainLLMCommand(
        args.train,
        args.deploy,
        args.inference,
        args.data_path,
        args.train_split,
        args.valid_split,
        args.text_column,
        args.model,
        args.learning_rate,
        args.num_train_epochs,
        args.train_batch_size,
        args.eval_batch_size,
        args.warmup_ratio,
        args.gradient_accumulation_steps,
        args.optimizer,
        args.scheduler,
        args.weight_decay,
        args.max_grad_norm,
        args.seed,
        args.add_eos_token,
        args.block_size,
        args.use_peft,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.training_type,
        args.train_on_inputs,
        args.logging_steps,
        args.project_name,
        args.evaluation_strategy,
        args.save_total_limit,
        args.save_strategy,
        args.auto_find_batch_size,
        args.fp16,
        args.push_to_hub,
        args.use_int8,
        args.model_max_length,
        args.repo_id,
        args.use_int4,
        args.trainer,
        args.target_modules,
    )


class RunAutoTrainLLMCommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        run_llm_parser = parser.add_parser(
            "llm",
            description="✨ Run AutoTrain LLM training/inference/deployment",
        )
        run_llm_parser.add_argument(
            "--train",
            help="Train the model",
            required=False,
            action="store_true",
        )
        run_llm_parser.add_argument(
            "--deploy",
            help="Deploy the model",
            required=False,
            action="store_true",
        )
        run_llm_parser.add_argument(
            "--inference",
            help="Run inference",
            required=False,
            action="store_true",
        )
        run_llm_parser.add_argument(
            "--data_path",
            help="Train dataset to use",
            required=False,
            type=str,
        )
        run_llm_parser.add_argument(
            "--train_split",
            help="Test dataset split to use",
            required=False,
            type=str,
            default="train",
        )
        run_llm_parser.add_argument(
            "--valid_split",
            help="Validation dataset split to use",
            required=False,
            type=str,
            default=None,
        )
        run_llm_parser.add_argument(
            "--text_column",
            help="Text column to use",
            required=False,
            type=str,
            default="text",
        )
        run_llm_parser.add_argument(
            "--model",
            help="Model to use",
            required=False,
            type=str,
        )
        run_llm_parser.add_argument(
            "--learning_rate",
            help="Learning rate to use",
            required=False,
            type=float,
            default=3e-5,
        )
        run_llm_parser.add_argument(
            "--num_train_epochs",
            help="Number of training epochs to use",
            required=False,
            type=int,
            default=1,
        )
        run_llm_parser.add_argument(
            "--train_batch_size",
            help="Training batch size to use",
            required=False,
            type=int,
            default=2,
        )
        run_llm_parser.add_argument(
            "--eval_batch_size",
            help="Evaluation batch size to use",
            required=False,
            type=int,
            default=4,
        )
        run_llm_parser.add_argument(
            "--warmup_ratio",
            help="Warmup proportion to use",
            required=False,
            type=float,
            default=0.1,
        )
        run_llm_parser.add_argument(
            "--gradient_accumulation_steps",
            help="Gradient accumulation steps to use",
            required=False,
            type=int,
            default=1,
        )
        run_llm_parser.add_argument(
            "--optimizer",
            help="Optimizer to use",
            required=False,
            type=str,
            default="adamw_torch",
        )
        run_llm_parser.add_argument(
            "--scheduler",
            help="Scheduler to use",
            required=False,
            type=str,
            default="linear",
        )
        run_llm_parser.add_argument(
            "--weight_decay",
            help="Weight decay to use",
            required=False,
            type=float,
            default=0.0,
        )
        run_llm_parser.add_argument(
            "--max_grad_norm",
            help="Max gradient norm to use",
            required=False,
            type=float,
            default=1.0,
        )
        run_llm_parser.add_argument(
            "--seed",
            help="Seed to use",
            required=False,
            type=int,
            default=42,
        )
        run_llm_parser.add_argument(
            "--add_eos_token",
            help="Add EOS token to use",
            required=False,
            action="store_true",
        )
        run_llm_parser.add_argument(
            "--block_size",
            help="Block size to use",
            required=False,
            type=int,
            default=-1,
        )
        run_llm_parser.add_argument(
            "--use_peft",
            help="Use PEFT to use",
            required=False,
            action="store_true",
        )
        run_llm_parser.add_argument(
            "--lora_r",
            help="Lora r to use",
            required=False,
            type=int,
            default=16,
        )
        run_llm_parser.add_argument(
            "--lora_alpha",
            help="Lora alpha to use",
            required=False,
            type=int,
            default=32,
        )
        run_llm_parser.add_argument(
            "--lora_dropout",
            help="Lora dropout to use",
            required=False,
            type=float,
            default=0.05,
        )
        run_llm_parser.add_argument(
            "--training_type",
            help="Training type to use",
            required=False,
            type=str,
            default="generic",
        )
        run_llm_parser.add_argument(
            "--train_on_inputs",
            help="Train on inputs to use",
            required=False,
            action="store_true",
        )
        run_llm_parser.add_argument(
            "--logging_steps",
            help="Logging steps to use",
            required=False,
            type=int,
            default=-1,
        )
        run_llm_parser.add_argument(
            "--project_name",
            help="Output directory",
            required=False,
            type=str,
        )
        run_llm_parser.add_argument(
            "--evaluation_strategy",
            help="Evaluation strategy to use",
            required=False,
            type=str,
            default="epoch",
        )
        run_llm_parser.add_argument(
            "--save_total_limit",
            help="Save total limit to use",
            required=False,
            type=int,
            default=1,
        )
        run_llm_parser.add_argument(
            "--save_strategy",
            help="Save strategy to use",
            required=False,
            type=str,
            default="epoch",
        )
        run_llm_parser.add_argument(
            "--auto_find_batch_size",
            help="Auto find batch size True/False",
            required=False,
            action="store_true",
        )
        run_llm_parser.add_argument(
            "--fp16",
            help="FP16 True/False",
            required=False,
            action="store_true",
        )
        run_llm_parser.add_argument(
            "--push_to_hub",
            help="Push to hub True/False",
            required=False,
            action="store_true",
        )
        run_llm_parser.add_argument(
            "--use_int8",
            help="Use int8 True/False",
            required=False,
            action="store_true",
        )
        run_llm_parser.add_argument(
            "--model_max_length",
            help="Model max length to use",
            required=False,
            type=int,
            default=1024,
        )
        run_llm_parser.add_argument(
            "--repo_id",
            help="Repo id for hugging face hub",
            required=False,
            type=str,
        )
        run_llm_parser.add_argument(
            "--use_int4",
            help="Use int4 True/False",
            required=False,
            action="store_true",
        )
        run_llm_parser.add_argument(
            "--trainer",
            help="Trainer type to use",
            required=False,
            type=str,
            default="default",
        )
        run_llm_parser.add_argument(
            "--target_modules",
            help="Target modules to use",
            required=False,
            type=str,
            default=None,
        )

        run_llm_parser.set_defaults(func=run_llm_command_factory)

    def __init__(
        self,
        train,
        deploy,
        inference,
        data_path,
        train_split,
        valid_split,
        text_column,
        model,
        learning_rate,
        num_train_epochs,
        train_batch_size,
        eval_batch_size,
        warmup_ratio,
        gradient_accumulation_steps,
        optimizer,
        scheduler,
        weight_decay,
        max_grad_norm,
        seed,
        add_eos_token,
        block_size,
        use_peft,
        lora_r,
        lora_alpha,
        lora_dropout,
        training_type,
        train_on_inputs,
        logging_steps,
        project_name,
        evaluation_strategy,
        save_total_limit,
        save_strategy,
        auto_find_batch_size,
        fp16,
        push_to_hub,
        use_int8,
        model_max_length,
        repo_id,
        use_int4,
        trainer,
        target_modules,
    ):
        self.train = train
        self.deploy = deploy
        self.inference = inference
        self.data_path = data_path
        self.train_split = train_split
        self.valid_split = valid_split
        self.text_column = text_column
        self.model = model
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.add_eos_token = add_eos_token
        self.block_size = block_size
        self.use_peft = use_peft
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.training_type = training_type
        self.train_on_inputs = train_on_inputs
        self.logging_steps = logging_steps
        self.project_name = project_name
        self.evaluation_strategy = evaluation_strategy
        self.save_total_limit = save_total_limit
        self.save_strategy = save_strategy
        self.auto_find_batch_size = auto_find_batch_size
        self.fp16 = fp16
        self.push_to_hub = push_to_hub
        self.use_int8 = use_int8
        self.model_max_length = model_max_length
        self.repo_id = repo_id
        self.use_int4 = use_int4
        self.trainer = trainer
        self.target_modules = target_modules

        if self.train:
            if self.project_name is None:
                raise ValueError("Project name must be specified")
            if self.data_path is None:
                raise ValueError("Data path must be specified")
            if self.model is None:
                raise ValueError("Model must be specified")
            if self.push_to_hub:
                if self.repo_id is None:
                    raise ValueError("Repo id must be specified for push to hub")

        if self.inference:
            tgi = TextGenerationInference(self.project_name, use_int4=self.use_int4, use_int8=self.use_int8)
            while True:
                prompt = input("User: ")
                if prompt == "exit()":
                    break
                print(f"Bot: {tgi.chat(prompt)}")

    def run(self):
        logger.info("Running LLM")
        logger.info(f"Train: {self.train}")
        if self.train:
            params = LLMTrainingParams(
                model_name=self.model,
                data_path=self.data_path,
                train_split=self.train_split,
                valid_split=self.valid_split,
                text_column=self.text_column,
                learning_rate=self.learning_rate,
                num_train_epochs=self.num_train_epochs,
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
                warmup_ratio=self.warmup_ratio,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                weight_decay=self.weight_decay,
                max_grad_norm=self.max_grad_norm,
                seed=self.seed,
                add_eos_token=self.add_eos_token,
                block_size=self.block_size,
                use_peft=self.use_peft,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                training_type=self.training_type,
                train_on_inputs=self.train_on_inputs,
                logging_steps=self.logging_steps,
                project_name=self.project_name,
                evaluation_strategy=self.evaluation_strategy,
                save_total_limit=self.save_total_limit,
                save_strategy=self.save_strategy,
                auto_find_batch_size=self.auto_find_batch_size,
                fp16=self.fp16,
                push_to_hub=self.push_to_hub,
                use_int8=self.use_int8,
                model_max_length=self.model_max_length,
                repo_id=self.repo_id,
                use_int4=self.use_int4,
                trainer=self.trainer,
                target_modules=self.target_modules,
            )
            train_llm(params)
