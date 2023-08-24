from functools import partial

import gradio as gr
import pandas as pd

from autotrain.apps import common
from autotrain.apps import utils as app_utils
from autotrain.dataset import AutoTrainDataset
from autotrain.languages import SUPPORTED_LANGUAGES
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
    col_map_label,
):
    if len(jobs_df) == 0:
        raise gr.Error("Please add at least one job.")
    task = "text_multi_class_classification"
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
        column_mapping={"text": col_map_text, "label": col_map_label},
        valid_data=validation_data,
        percent_valid=None,  # TODO: add to UI
        convert_to_class_label=True,
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
        gr.Markdown("### 🚀 Text Classification")
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
                        col_map_target = gr.Dropdown(
                            label="Target Column",
                            choices=[],
                            visible=True,
                            interactive=True,
                            elem_id="col_map_target",
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
                    hyp_use_fp16 = gr.Checkbox(
                        label="FP16",
                        value=True,
                        visible=True,
                        interactive=True,
                        elem_id="hyp_use_fp16",
                    )

            with gr.Column():
                with gr.Group():
                    param_choice = gr.Dropdown(
                        label="Parameter Choice",
                        choices=["Manual", "AutoTrain"],
                        value="Manual",
                        visible=True,
                        interactive=True,
                        elem_id="param_choice",
                    )
                    with gr.Row():
                        hyp_language = gr.Dropdown(
                            label="Language",
                            choices=SUPPORTED_LANGUAGES,
                            value="en",
                            visible=False,
                            interactive=False,
                            elem_id="hyp_language",
                        )
                    with gr.Row():
                        hyp_num_jobs = gr.Number(
                            label="Num Jobs",
                            value=5,
                            visible=False,
                            interactive=False,
                            elem_id="hyp_num_jobs",
                            precision=0,
                        )
                    with gr.Row():
                        hyp_lr = gr.Number(
                            label="Learning Rate",
                            value=5e-5,
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
                        hyp_max_seq_length = gr.Number(
                            label="Max Seq Length",
                            value=512,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_max_seq_length",
                        )
                        hyp_batch_size = gr.Number(
                            label="Batch Size",
                            value=8,
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
                return [
                    gr.Dropdown.update(visible=True, interactive=False, choices=[], label="`text` column"),
                    gr.Dropdown.update(
                        visible=True,
                        interactive=False,
                        choices=[],
                        label="`target` column",
                    ),
                ]
            return [
                gr.Dropdown.update(
                    visible=True,
                    interactive=True,
                    choices=data_cols,
                    label="`text` column",
                    value=data_cols[0],
                ),
                gr.Dropdown.update(
                    visible=True,
                    interactive=True,
                    choices=data_cols,
                    label="`target` column",
                    value=data_cols[1],
                ),
            ]

        training_data.change(
            _update_col_map,
            inputs=training_data,
            outputs=[col_map_text, col_map_target],
        )

        hyperparameters = [
            hyp_scheduler,
            hyp_optimizer,
            hyp_lr,
            hyp_epochs,
            hyp_max_seq_length,
            hyp_batch_size,
            hyp_warmup_ratio,
            hyp_weight_decay,
            hyp_gradient_accumulation,
            hyp_language,
            hyp_num_jobs,
            hyp_use_fp16,
        ]

        model_choice.change(
            app_utils.handle_model_choice_change,
            inputs=model_choice,
            outputs=[jobs_df, param_choice, autotrain_backend],
        )

        def _handle_param_choice_change(components):
            hyperparam_visibility = {}
            if components[param_choice] == "AutoTrain":
                for _hyperparameter in hyperparameters:
                    if _hyperparameter.elem_id in ["hyp_num_jobs", "hyp_language"]:
                        hyperparam_visibility[_hyperparameter.elem_id] = True
                    else:
                        hyperparam_visibility[_hyperparameter.elem_id] = False
            else:
                for _hyperparameter in hyperparameters:
                    if _hyperparameter.elem_id in ["hyp_num_jobs", "hyp_language"]:
                        hyperparam_visibility[_hyperparameter.elem_id] = False
                    else:
                        hyperparam_visibility[_hyperparameter.elem_id] = True
            op = [
                h.update(
                    interactive=hyperparam_visibility.get(h.elem_id, False),
                    visible=hyperparam_visibility.get(h.elem_id, False),
                )
                for h in hyperparameters
            ]
            op.append(jobs_df.update(value=pd.DataFrame(columns=[0]), visible=False, interactive=False))
            return op

        param_choice.change(
            _handle_param_choice_change,
            inputs=set([param_choice]),
            outputs=hyperparameters + [jobs_df],
        )

        def _add_job(components):
            try:
                _ = pd.read_csv(components[training_data][0].name, nrows=2).columns.tolist()
            except TypeError:
                raise gr.Error("Please upload training data first.")
            if len(str(components[col_map_text].strip())) == 0:
                raise gr.Error("Text column cannot be empty.")
            if len(str(components[col_map_target].strip())) == 0:
                raise gr.Error("Target column cannot be empty.")
            if components[col_map_text] == components[col_map_target]:
                raise gr.Error("Text and Target column cannot be the same.")
            if components[param_choice] == "AutoTrain" and components[autotrain_backend] != "AutoTrain":
                raise gr.Error("AutoTrain param choice is only available with AutoTrain backend.")

            _training_params = {}
            if components[param_choice] == "AutoTrain":
                for _hyperparameter in hyperparameters:
                    if _hyperparameter.elem_id in ["hyp_num_jobs", "hyp_language"]:
                        _training_params[_hyperparameter.elem_id] = components[_hyperparameter]
            else:
                for _hyperparameter in hyperparameters:
                    if _hyperparameter.elem_id not in ["hyp_num_jobs", "hyp_language"]:
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
                    col_map_target,
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
                col_map_target,
            ],
            outputs=output_md,
        )

        demo.load(
            app_utils._update_project_name,
            outputs=project_name,
        ).then(
            partial(app_utils._update_hub_model_choices, task="text_multi_class_classification"),
            outputs=model_choice,
        )

    return demo


if __name__ == "__main__":
    demo = main()
    demo.launch()
