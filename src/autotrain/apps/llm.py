from functools import partial

import gradio as gr
import pandas as pd

from autotrain.apps import common
from autotrain.apps import utils as app_utils
from autotrain.dataset import AutoTrainDataset
from autotrain.project import AutoTrainProject


def start_training(
    jobs_df,
    model_choice,
    training_data,
    validation_data,
    project_name,
    autotrain_username,
    user_token,
    col_map_text,
):
    if len(jobs_df) == 0:
        raise gr.Error("Please add at least one job.")
    task = "lm_training"
    training_data = [f.name for f in training_data]
    if validation_data is None:
        validation_data = []
    else:
        validation_data = [f.name for f in validation_data]
    dset = AutoTrainDataset(
        train_data=training_data,
        task=task,
        token=user_token,
        project_name=project_name,
        username=autotrain_username,
        column_mapping={"text": col_map_text},
        valid_data=validation_data,
        percent_valid=None,  # TODO: add to UI
    )
    dset.prepare()
    project = AutoTrainProject(dataset=dset, job_params=jobs_df)
    ids = project.create()
    return gr.Markdown.update(
        value=f"Training started for {len(ids)} jobs. You can view the status of your jobs at ids: {', '.join(ids)}",
        visible=True,
    )


def main():
    with gr.Blocks(theme=app_utils.THEME) as demo:
        gr.Markdown("### 🚀 LLM Finetuning")
        user_token, valid_can_pay, who_is_training = common.user_validation()
        autotrain_username, project_name, model_choice, autotrain_backend = common.base_components(who_is_training)
        with gr.Row():
            training_data, validation_data = common.train_valid_components()
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        col_map_text = gr.Dropdown(
                            label="Text Column",
                            choices=[],
                            visible=True,
                            interactive=True,
                            elem_id="col_map_text",
                        )
                    with gr.Row():
                        hyp_scheduler = gr.Dropdown(
                            label="Scheduler",
                            choices=["cosine", "linear", "constant"],
                            value="linear",
                            visible=True,
                            interactive=True,
                            elem_id="hyp_scheduler",
                        )
                        hyp_optimizer = gr.Dropdown(
                            label="Optimizer",
                            choices=["adamw_torch", "adamw_hf", "sgd", "adafactor", "adagrad"],
                            value="adamw_torch",
                            visible=True,
                            interactive=True,
                            elem_id="hyp_optimizer",
                        )
                    with gr.Row():
                        hyp_use_peft = gr.Checkbox(
                            label="PEFT",
                            value=True,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_use_peft",
                        )
                        hyp_use_fp16 = gr.Checkbox(
                            label="FP16",
                            value=True,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_use_fp16",
                        )
                    with gr.Row():
                        hyp_int4_8 = gr.Dropdown(
                            label="Int4/8",
                            choices=["none", "int4", "int8"],
                            value="int4",
                            visible=True,
                            interactive=True,
                            elem_id="hyp_int4_8",
                        )
                        hyp_lora_r = gr.Number(
                            label="LoRA R",
                            value=16,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_lora_r",
                            precision=0,
                        )
                    with gr.Row():
                        hyp_lora_alpha = gr.Number(
                            label="LoRA Alpha",
                            value=32,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_lora_alpha",
                            precision=0,
                        )
                        hyp_lora_dropout = gr.Number(
                            label="LoRA Dropout",
                            value=0.1,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_lora_dropout",
                        )

            with gr.Column():
                with gr.Group():
                    param_choice = gr.Dropdown(
                        label="Parameter Choice",
                        choices=["Manual"],
                        value="Manual",
                        visible=True,
                        interactive=True,
                        elem_id="param_choice",
                    )
                    with gr.Row():
                        hyp_lr = gr.Number(
                            label="Learning Rate",
                            value=2e-4,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_lr",
                        )
                        hyp_epochs = gr.Number(
                            label="Epochs",
                            value=3,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_epochs",
                        )
                    with gr.Row():
                        hyp_block_size = gr.Number(
                            label="Block Size",
                            value=1024,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_block_size",
                        )
                        hyp_batch_size = gr.Number(
                            label="Batch Size",
                            value=2,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_batch_size",
                        )
                    with gr.Row():
                        hyp_warmup_ratio = gr.Number(
                            label="Warmup Ratio",
                            value=0.1,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_warmup_ratio",
                        )
                        hyp_weight_decay = gr.Number(
                            label="Weight Decay",
                            value=0.01,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_weight_decay",
                        )
                    with gr.Row():
                        hyp_gradient_accumulation = gr.Number(
                            label="Grad Acc Steps",
                            value=1,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_gradient_accumulation",
                        )
                        hyp_trainer = gr.Dropdown(
                            label="Trainer Type",
                            choices=["Default", "SFT"],
                            value="SFT",
                            visible=True,
                            interactive=True,
                            elem_id="hyp_trainer",
                        )

        with gr.Row():
            add_job_button = gr.Button(value="Add Job", elem_id="add_job_button")
            clear_jobs_button = gr.Button(value="Clear Jobs", elem_id="clear_jobs_button")
            start_training_button = gr.Button(value="Start Training", elem_id="start_training_button")

        output_md = gr.Markdown(
            value="WARNING: Clicking `Start Training` will incur costs!", visible=True, interactive=False
        )
        jobs_df = gr.DataFrame(visible=False, interactive=False, value=pd.DataFrame())

        def _update_col_map(training_data):
            try:
                data_cols = pd.read_csv(training_data[0].name, nrows=2).columns.tolist()
            except TypeError:
                return (gr.Dropdown.update(visible=True, interactive=False, choices=[], label="`text` column"),)
            return gr.Dropdown.update(
                visible=True,
                interactive=True,
                choices=data_cols,
                label="`text` column",
                value=data_cols[0],
            )

        training_data.change(
            _update_col_map,
            inputs=training_data,
            outputs=col_map_text,
        )

        hyperparameters = [
            hyp_scheduler,
            hyp_optimizer,
            hyp_use_peft,
            hyp_use_fp16,
            hyp_int4_8,
            hyp_lora_r,
            hyp_lora_alpha,
            hyp_lora_dropout,
            hyp_lr,
            hyp_epochs,
            hyp_block_size,
            hyp_batch_size,
            hyp_warmup_ratio,
            hyp_weight_decay,
            hyp_gradient_accumulation,
            hyp_trainer,
        ]

        model_choice.change(
            app_utils.handle_model_choice_change,
            inputs=model_choice,
            outputs=[jobs_df, param_choice, autotrain_backend],
        )

        def _add_job(components):
            try:
                _ = pd.read_csv(components[training_data][0].name, nrows=2).columns.tolist()
            except TypeError:
                raise gr.Error("Please upload training data first.")
            if len(str(components[col_map_text].strip())) == 0:
                raise gr.Error("Text column cannot be empty.")
            if components[param_choice] == "AutoTrain" and components[autotrain_backend] != "AutoTrain":
                raise gr.Error("AutoTrain param choice is only available with AutoTrain backend.")

            _training_params = {}
            for _hyperparameter in hyperparameters:
                _training_params[_hyperparameter.elem_id] = components[_hyperparameter]

            _training_params_df = app_utils.fetch_training_params_df(
                components[param_choice],
                components[jobs_df],
                _training_params,
                components[model_choice],
                components[autotrain_backend],
            )
            return gr.DataFrame.update(value=_training_params_df, visible=True, interactive=False)

        add_job_button.click(
            _add_job,
            inputs=set(
                [
                    training_data,
                    param_choice,
                    autotrain_backend,
                    col_map_text,
                    jobs_df,
                    model_choice,
                    autotrain_backend,
                ]
                + hyperparameters
            ),
            outputs=jobs_df,
        )

        clear_jobs_button.click(
            app_utils.clear_jobs,
            inputs=jobs_df,
            outputs=jobs_df,
        )

        start_training_button.click(
            start_training,
            inputs=[
                jobs_df,
                model_choice,
                training_data,
                validation_data,
                project_name,
                autotrain_username,
                user_token,
                col_map_text,
            ],
            outputs=output_md,
        )

        demo.load(
            app_utils._update_project_name,
            outputs=project_name,
        ).then(
            partial(app_utils._update_hub_model_choices, task="lm_training"),
            outputs=model_choice,
        )

    return demo


if __name__ == "__main__":
    demo = main()
    demo.launch()
