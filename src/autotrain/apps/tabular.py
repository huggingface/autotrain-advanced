from functools import partial

import gradio as gr
import pandas as pd

from autotrain.apps import common
from autotrain.apps import utils as app_utils
from autotrain.dataset import AutoTrainDataset
from autotrain.project import AutoTrainProject


ALLOWED_MODELS = [
    "xgboost",
    "random_forest",
    "ridge",
    "logistic_regression",
    "svm",
    "extra_trees",
    "gradient_boosting",
    "adaboost",
    "decision_tree",
    "knn",
]


def start_training(
    jobs_df,
    model_choice,
    training_data,
    validation_data,
    project_name,
    autotrain_username,
    user_token,
    col_map_id,
    col_map_label,
    task,
):
    if len(jobs_df) == 0:
        raise gr.Error("Please add at least one job.")
    if isinstance(col_map_label, str):
        col_map_label = [col_map_label]
    if task == "classification" and len(col_map_label) > 1:
        task = "tabular_multi_label_classification"
    elif task == "classification" and len(col_map_label) == 1:
        task = "tabular_multi_class_classification"
    elif task == "regression" and len(col_map_label) > 1:
        task = "tabular_multi_column_regression"
    elif task == "regression" and len(col_map_label) == 1:
        task = "tabular_single_column_regression"
    else:
        raise gr.Error("Please select a valid task.")

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
        column_mapping={"id": col_map_id, "label": col_map_label},
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
        gr.Markdown("### 🚀 Tabular Classification / Regression")
        user_token, valid_can_pay, who_is_training = common.user_validation()
        autotrain_username, project_name, model_choice, autotrain_backend = common.base_components(who_is_training)
        model_choice.update(label="", visible=False, interactive=False)
        with gr.Row():
            training_data, validation_data = common.train_valid_components()
            with gr.Column():
                with gr.Group():
                    col_map_id = gr.Dropdown(
                        label="`id` column",
                        choices=[],
                        visible=True,
                        interactive=True,
                        elem_id="col_map_id",
                    )
                    col_map_target = gr.Dropdown(
                        label="`target` column(s)",
                        choices=[],
                        visible=True,
                        interactive=True,
                        elem_id="col_map_target",
                        multiselect=True,
                    )
                    with gr.Row():
                        hyp_task = gr.Dropdown(
                            label="Task",
                            choices=["classification", "regression"],
                            value="classification",
                            visible=True,
                            interactive=True,
                            elem_id="hyp_task",
                        )

            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        param_choice = gr.Dropdown(
                            label="Param Choice",
                            choices=["Manual"],
                            value="Manual",
                            visible=True,
                            interactive=True,
                            elem_id="param_choice",
                        )
                    hyp_model = gr.Dropdown(
                        label="Model",
                        choices=ALLOWED_MODELS,
                        value=ALLOWED_MODELS[0],
                        visible=True,
                        interactive=True,
                        elem_id="hyp_model",
                    )
                    hyp_categorial_imputer = gr.Dropdown(
                        label="Categorical Imputer",
                        choices=["most_frequent", "none"],
                        value="none",
                        visible=True,
                        interactive=True,
                        elem_id="hyp_categorical_imputer",
                    )
                    with gr.Row():
                        hyp_numerical_imputer = gr.Dropdown(
                            label="Numerical Imputer",
                            choices=["mean", "median", "most_frequent", "none"],
                            value="mean",
                            visible=True,
                            interactive=True,
                            elem_id="hyp_numerical_imputer",
                        )
                        hyp_numeric_scaler = gr.Dropdown(
                            label="Numeric Scaler",
                            choices=["standard", "minmax", "normal", "robust", "none"],
                            value="standard",
                            visible=True,
                            interactive=True,
                            elem_id="hyp_numeric_scaler",
                        )
                    with gr.Row():
                        hyp_num_trials = gr.Number(
                            label="Num Trials",
                            value=100,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_num_trials",
                            precision=0,
                        )
                        hyp_time_limit = gr.Number(
                            label="Time Limit",
                            value=3600,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_time_limit",
                            precision=0,
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
                    gr.Dropdown.update(visible=True, interactive=False, choices=[], label="`id` column"),
                    gr.Dropdown.update(
                        visible=True,
                        interactive=False,
                        choices=[],
                        label="`target` column(s)",
                    ),
                ]
            return [
                gr.Dropdown.update(
                    visible=True,
                    interactive=True,
                    choices=[" "] + data_cols,
                    label="`id` column",
                    value="",
                ),
                gr.Dropdown.update(
                    visible=True,
                    interactive=True,
                    choices=data_cols,
                    label="`target` column(s)",
                    value=data_cols[1],
                ),
            ]

        training_data.change(
            _update_col_map,
            inputs=training_data,
            outputs=[col_map_id, col_map_target],
        )

        hyperparameters = [
            hyp_task,
            hyp_model,
            hyp_categorial_imputer,
            hyp_numerical_imputer,
            hyp_numeric_scaler,
            hyp_num_trials,
            hyp_time_limit,
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
                    hyperparam_visibility[_hyperparameter.elem_id] = False
            else:
                for _hyperparameter in hyperparameters:
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
            if len(components[col_map_target]) == 0:
                raise gr.Error("Target column cannot be empty.")
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
                hide_model_param=True,
            )
            return gr.DataFrame.update(value=_training_params_df, visible=True, interactive=False)

        add_job_button.click(
            _add_job,
            inputs=set(
                [
                    training_data,
                    param_choice,
                    autotrain_backend,
                    col_map_id,
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
                col_map_id,
                col_map_target,
                hyp_task,
            ],
            outputs=output_md,
        )

        demo.load(
            app_utils._update_project_name,
            outputs=project_name,
        ).then(
            partial(app_utils._update_hub_model_choices, task="tabular"),
            outputs=model_choice,
        )

    return demo


if __name__ == "__main__":
    demo = main()
    demo.launch()
