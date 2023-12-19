import io
import os

from autotrain.dataset import AutoTrainDataset, AutoTrainDreamboothDataset


def tabular_munge_data(params, local):
    if isinstance(params.target_columns, str):
        col_map_label = [params.target_columns]
    else:
        col_map_label = params.target_columns
    task = params.task
    if task == "classification" and len(col_map_label) > 1:
        task = "tabular_multi_label_classification"
    elif task == "classification" and len(col_map_label) == 1:
        task = "tabular_multi_class_classification"
    elif task == "regression" and len(col_map_label) > 1:
        task = "tabular_multi_column_regression"
    elif task == "regression" and len(col_map_label) == 1:
        task = "tabular_single_column_regression"
    else:
        raise Exception("Please select a valid task.")

    train_data_path = f"{params.data_path}/{params.train_split}.csv"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.csv"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            task=task,
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping={"id": params.col_map_id, "label": col_map_label},
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            percent_valid=None,  # TODO: add to UI
            local=local,
        )
        dset.prepare()
        if local:
            return f"{params.project_name}/autotrain-data"
        return f"{params.username}/autotrain-data-{params.project_name}"

    return params.data_path


def llm_munge_data(params, local):
    train_data_path = f"{params.data_path}/{params.train_split}.csv"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.csv"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        col_map = {"text": params.text_column}
        if params.rejected_text_column is not None:
            col_map["rejected_text"] = params.rejected_text_column
        if params.prompt_column is not None:
            col_map["prompt"] = params.prompt_column
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            task="lm_training",
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping=col_map,
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            percent_valid=None,  # TODO: add to UI
            local=local,
        )
        dset.prepare()
        if local:
            return f"{params.project_name}/autotrain-data"
        return f"{params.username}/autotrain-data-{params.project_name}"

    return params.data_path


def seq2seq_munge_data(params, local):
    train_data_path = f"{params.data_path}/{params.train_split}.csv"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.csv"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            task="seq2seq",
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping={"text": params.text_column, "label": params.target_column},
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            percent_valid=None,  # TODO: add to UI
            local=local,
        )
        dset.prepare()
        if local:
            return f"{params.project_name}/autotrain-data"
        return f"{params.username}/autotrain-data-{params.project_name}"

    return params.data_path


def text_clf_munge_data(params, local):
    train_data_path = f"{params.data_path}/{params.train_split}.csv"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.csv"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            task="text_multi_class_classification",
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping={"text": params.text_column, "label": params.target_column},
            percent_valid=None,  # TODO: add to UI
            local=local,
            convert_to_class_label=True,
        )
        dset.prepare()
        if local:
            return f"{params.project_name}/autotrain-data"
        return f"{params.username}/autotrain-data-{params.project_name}"

    return params.data_path


def img_clf_munge_data(params, local):
    train_data_path = f"{params.data_path}/{params.train_split}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}"
    else:
        valid_data_path = None
    if os.path.isdir(train_data_path) or os.path.isdir(valid_data_path):
        raise Exception("Image classification is not yet supported for local datasets using the CLI. Please use UI.")
    return params.data_path


def dreambooth_munge_data(params, local):
    # check if params.image_path is a directory
    if os.path.isdir(params.image_path):
        training_data = [os.path.join(params.image_path, f) for f in os.listdir(params.image_path)]
        training_data = [io.BytesIO(open(f, "rb").read()) for f in training_data]
        dset = AutoTrainDreamboothDataset(
            concept_images=training_data,
            concept_name=params.prompt,
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            local=local,
        )
        dset.prepare()
        if local:
            return f"{params.project_name}/autotrain-data"
        return f"{params.username}/autotrain-data-{params.project_name}"
    return params.image_path
