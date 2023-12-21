import argparse
import json
import os
from functools import partial

import joblib
import numpy as np
import optuna
import pandas as pd
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi
from sklearn import pipeline, preprocessing
from sklearn.compose import ColumnTransformer

from autotrain import logger
from autotrain.trainers.common import monitor, pause_space, remove_autotrain_data, save_training_params
from autotrain.trainers.tabular import utils
from autotrain.trainers.tabular.params import TabularParams


def parse_args():
    # get training_config.json from the end user
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()


def optimize(trial, model_name, xtrain, xvalid, ytrain, yvalid, eval_metric, task, preprocessor):
    if isinstance(trial, dict):
        params = trial
    else:
        params = utils.get_params(trial, model_name, task)
    labels = None
    if task == "multi_class_classification":
        labels = np.unique(ytrain)
    metrics = utils.TabularMetrics(sub_task=task, labels=labels)

    if task in ("binary_classification", "multi_class_classification", "single_column_regression"):
        ytrain = ytrain.ravel()
        yvalid = yvalid.ravel()

    if preprocessor is not None:
        try:
            xtrain = preprocessor.fit_transform(xtrain)
            xvalid = preprocessor.transform(xvalid)
        except ValueError:
            logger.info("Preprocessing failed, using nan_to_num")
            train_cols = xtrain.columns.tolist()
            valid_cols = xvalid.columns.tolist()
            xtrain = np.nan_to_num(xtrain)
            xvalid = np.nan_to_num(xvalid)
            # convert back to dataframe
            xtrain = pd.DataFrame(xtrain, columns=train_cols)
            xvalid = pd.DataFrame(xvalid, columns=valid_cols)
            xtrain = preprocessor.fit_transform(xtrain)
            xvalid = preprocessor.transform(xvalid)

    if model_name == "xgboost":
        params["eval_metric"] = eval_metric

    _model = utils.TabularModel(model_name, preprocessor=None, sub_task=task, params=params)
    model = _model.pipeline
    models = []
    if task in ("multi_label_classification", "multi_column_regression"):
        # also multi_column_regression
        ypred = []
        models = [model] * ytrain.shape[1]
        for idx, _m in enumerate(models):
            if model_name == "xgboost":
                _m.fit(
                    xtrain,
                    ytrain[:, idx],
                    model__eval_set=[(xvalid, yvalid[:, idx])],
                    model__verbose=False,
                )
            else:
                _m.fit(xtrain, ytrain[:, idx])
            if task == "multi_column_regression":
                ypred_temp = _m.predict(xvalid)
            else:
                if _model.use_predict_proba:
                    ypred_temp = _m.predict_proba(xvalid)[:, 1]
                else:
                    ypred_temp = _m.predict(xvalid)
            ypred.append(ypred_temp)
        ypred = np.column_stack(ypred)

    else:
        models = [model]
        if model_name == "xgboost":
            model.fit(
                xtrain,
                ytrain,
                model__eval_set=[(xvalid, yvalid)],
                model__verbose=False,
            )
        else:
            models[0].fit(xtrain, ytrain)

        if _model.use_predict_proba:
            ypred = models[0].predict_proba(xvalid)
        else:
            ypred = models[0].predict(xvalid)

        if task == "multi_class_classification":
            if ypred.reshape(xvalid.shape[0], -1).shape[1] != len(labels):
                ypred_ohe = np.zeros((xvalid.shape[0], len(labels)))
                ypred_ohe[np.arange(xvalid.shape[0]), ypred] = 1
                ypred = ypred_ohe

        if task == "binary_classification":
            if ypred.reshape(xvalid.shape[0], -1).shape[1] != 2:
                ypred = np.column_stack([1 - ypred, ypred])

    # calculate metric
    metric_dict = metrics.calculate(yvalid, ypred)

    # change eval_metric key to loss
    if eval_metric in metric_dict:
        metric_dict["loss"] = metric_dict[eval_metric]

    logger.info(f"Metrics: {metric_dict}")
    if isinstance(trial, dict):
        return models, preprocessor, metric_dict
    return metric_dict["loss"]


@monitor
def train(config):
    if isinstance(config, dict):
        config = TabularParams(**config)

    if config.repo_id is None and config.username is not None:
        config.repo_id = f"{config.username}/{config.project_name}"

    logger.info("Starting training...")
    logger.info(f"Training config: {config}")

    train_data = None
    valid_data = None
    if config.data_path == f"{config.project_name}/autotrain-data":
        logger.info("loading dataset from disk")
        train_data = load_from_disk(config.data_path)[config.train_split]
    else:
        train_data = load_dataset(
            config.data_path,
            split=config.train_split,
            token=config.token,
        )
    train_data = train_data.to_pandas()

    if config.valid_split is not None:
        if config.data_path == f"{config.project_name}/autotrain-data":
            logger.info("loading dataset from disk")
            valid_data = load_from_disk(config.data_path)[config.valid_split]
        else:
            valid_data = load_dataset(
                config.data_path,
                split=config.valid_split,
                token=config.token,
            )
        valid_data = valid_data.to_pandas()

    if valid_data is None:
        raise Exception("valid_data is None. Please provide a valid_split for tabular training.")

    # determine which columns are categorical
    if config.categorical_columns is None:
        config.categorical_columns = utils.get_categorical_columns(train_data)
    if config.numerical_columns is None:
        config.numerical_columns = utils.get_numerical_columns(train_data)

    _id_target_cols = (
        [config.id_column] + config.target_columns if config.id_column is not None else config.target_columns
    )
    config.numerical_columns = [c for c in config.numerical_columns if c not in _id_target_cols]
    config.categorical_columns = [c for c in config.categorical_columns if c not in _id_target_cols]

    useful_columns = config.categorical_columns + config.numerical_columns

    logger.info(f"Categorical columns: {config.categorical_columns}")
    logger.info(f"Numerical columns: {config.numerical_columns}")

    # convert object columns to categorical
    for col in config.categorical_columns:
        train_data[col] = train_data[col].astype("category")
        valid_data[col] = valid_data[col].astype("category")

    logger.info(f"Useful columns: {useful_columns}")

    target_encoders = {}
    if config.task == "classification":
        for target_column in config.target_columns:
            target_encoder = preprocessing.LabelEncoder()
            target_encoder.fit(train_data[target_column])
            target_encoders[target_column] = target_encoder

    # encode target columns in train and valid data
    for k, v in target_encoders.items():
        train_data.loc[:, k] = v.transform(train_data[k])
        valid_data.loc[:, k] = v.transform(valid_data[k])

    numeric_transformer = "passthrough"
    categorical_transformer = "passthrough"
    transformers = []
    preprocessor = None

    numeric_steps = []
    imputer = utils.get_imputer(config.numerical_imputer)
    scaler = utils.get_scaler(config.numeric_scaler)
    if imputer is not None:
        numeric_steps.append(("num_imputer", imputer))
    if scaler is not None:
        numeric_steps.append(("num_scaler", scaler))

    if len(numeric_steps) > 0:
        numeric_transformer = pipeline.Pipeline(numeric_steps)
        transformers.append(("numeric", numeric_transformer, config.numerical_columns))

    categorical_steps = []
    imputer = utils.get_imputer(config.categorical_imputer)
    if imputer is not None:
        categorical_steps.append(("cat_imputer", imputer))

    if len(config.categorical_columns) > 0:
        if config.model in ("xgboost", "lightgbm", "randomforest", "catboost", "extratrees"):
            categorical_steps.append(
                (
                    "cat_encoder",
                    preprocessing.OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        categories="auto",
                        unknown_value=np.nan,
                    ),
                )
            )
        else:
            categorical_steps.append(
                (
                    "cat_encoder",
                    preprocessing.OneHotEncoder(handle_unknown="ignore"),
                )
            )

    if len(categorical_steps) > 0:
        categorical_transformer = pipeline.Pipeline(categorical_steps)
        transformers.append(("categorical", categorical_transformer, config.categorical_columns))

    if len(transformers) > 0:
        preprocessor = ColumnTransformer(transformers=transformers, verbose=True, n_jobs=-1)
        logger.info(f"Preprocessor: {preprocessor}")

    xtrain = train_data[useful_columns].reset_index(drop=True)
    xvalid = valid_data[useful_columns].reset_index(drop=True)

    ytrain = train_data[config.target_columns].values
    yvalid = valid_data[config.target_columns].values

    # determine sub_task
    if config.task == "classification":
        if len(target_encoders) == 1:
            if len(target_encoders[config.target_columns[0]].classes_) == 2:
                sub_task = "binary_classification"
            else:
                sub_task = "multi_class_classification"
        else:
            sub_task = "multi_label_classification"
    else:
        if len(config.target_columns) > 1:
            sub_task = "multi_column_regression"
        else:
            sub_task = "single_column_regression"

    eval_metric, direction = utils.get_metric_direction(sub_task)

    logger.info(f"Sub task: {sub_task}")

    args = {
        "model_name": config.model,
        "xtrain": xtrain,
        "xvalid": xvalid,
        "ytrain": ytrain,
        "yvalid": yvalid,
        "eval_metric": eval_metric,
        "task": sub_task,
        "preprocessor": preprocessor,
    }

    optimize_func = partial(optimize, **args)
    study = optuna.create_study(direction=direction, study_name="AutoTrain")
    study.optimize(optimize_func, n_trials=config.num_trials, timeout=config.time_limit)
    best_params = study.best_params

    logger.info(f"Best params: {best_params}")
    best_models, best_preprocessors, best_metrics = optimize(best_params, **args)

    models = (
        [pipeline.Pipeline([("preprocessor", best_preprocessors), ("model", m)]) for m in best_models]
        if best_preprocessors is not None
        else best_models
    )

    joblib.dump(
        models[0] if len(models) == 1 else models,
        os.path.join(config.project_name, "model.joblib"),
    )
    joblib.dump(target_encoders, os.path.join(config.project_name, "target_encoders.joblib"))

    model_card = utils.create_model_card(config, sub_task, best_params, best_metrics)

    if model_card is not None:
        with open(os.path.join(config.project_name, "README.md"), "w") as fp:
            fp.write(f"{model_card}")

    # remove token key from training_params.json located in output directory
    # first check if file exists
    if os.path.exists(f"{config.project_name}/training_params.json"):
        training_params = json.load(open(f"{config.project_name}/training_params.json"))
        training_params.pop("token")
        json.dump(training_params, open(f"{config.project_name}/training_params.json", "w"))

    # save model card to output directory as README.md
    with open(f"{config.project_name}/README.md", "w") as f:
        f.write(model_card)

    if config.push_to_hub:
        remove_autotrain_data(config)
        save_training_params(config)
        logger.info("Pushing model to hub...")
        api = HfApi(token=config.token)
        api.create_repo(repo_id=config.repo_id, repo_type="model", private=True)
        api.upload_folder(folder_path=config.project_name, repo_id=config.repo_id, repo_type="model")

    pause_space(config)


if __name__ == "__main__":
    args = parse_args()
    training_config = json.load(open(args.training_config))
    config = TabularParams(**training_config)
    train(config)
