import copy
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import List, Optional

import numpy as np
from sklearn import ensemble, impute, linear_model
from sklearn import metrics as skmetrics
from sklearn import naive_bayes, neighbors, pipeline, preprocessing, svm, tree
from xgboost import XGBClassifier, XGBRegressor


MARKDOWN = """
---
tags:
- autotrain
- tabular
- {task}
- tabular-{task}
datasets:
- {dataset}
---

# Model Trained Using AutoTrain

- Problem type: Tabular {task}

## Validation Metrics

{metrics}

## Best Params

{params}

## Usage

```python
import json
import joblib
import pandas as pd

model = joblib.load('model.joblib')
config = json.load(open('config.json'))

features = config['features']

# data = pd.read_csv("data.csv")
data = data[features]

predictions = model.predict(data)  # or model.predict_proba(data)

# predictions can be converted to original labels using label_encoders.pkl

```
"""

_MODELS: dict = defaultdict(dict)
_MODELS["xgboost"]["classification"] = XGBClassifier
_MODELS["xgboost"]["regression"] = XGBRegressor
_MODELS["logistic_regression"]["classification"] = linear_model.LogisticRegression
_MODELS["logistic_regression"]["regression"] = linear_model.LogisticRegression
_MODELS["random_forest"]["classification"] = ensemble.RandomForestClassifier
_MODELS["random_forest"]["regression"] = ensemble.RandomForestRegressor
_MODELS["extra_trees"]["classification"] = ensemble.ExtraTreesClassifier
_MODELS["extra_trees"]["regression"] = ensemble.ExtraTreesRegressor
_MODELS["gradient_boosting"]["classification"] = ensemble.GradientBoostingClassifier
_MODELS["gradient_boosting"]["regression"] = ensemble.GradientBoostingRegressor
_MODELS["adaboost"]["classification"] = ensemble.AdaBoostClassifier
_MODELS["adaboost"]["regression"] = ensemble.AdaBoostRegressor
_MODELS["ridge"]["classification"] = linear_model.RidgeClassifier
_MODELS["ridge"]["regression"] = linear_model.Ridge
_MODELS["svm"]["classification"] = svm.LinearSVC
_MODELS["svm"]["regression"] = svm.LinearSVR
_MODELS["decision_tree"]["classification"] = tree.DecisionTreeClassifier
_MODELS["decision_tree"]["regression"] = tree.DecisionTreeRegressor
_MODELS["lasso"]["regression"] = linear_model.Lasso
_MODELS["linear_regression"]["regression"] = linear_model.LinearRegression
_MODELS["naive_bayes"]["classification"] = naive_bayes.GaussianNB
_MODELS["knn"]["classification"] = neighbors.KNeighborsClassifier
_MODELS["knn"]["regression"] = neighbors.KNeighborsRegressor

CLASSIFICATION_TASKS = ("binary_classification", "multi_class_classification", "multi_label_classification")
REGRESSION_TASKS = ("single_column_regression", "multi_column_regression")


@dataclass
class TabularMetrics:
    sub_task: str
    labels: Optional[List] = None

    def __post_init__(self):
        if self.sub_task == "binary_classification":
            self.valid_metrics = {
                "auc": skmetrics.roc_auc_score,
                "logloss": skmetrics.log_loss,
                "f1": skmetrics.f1_score,
                "accuracy": skmetrics.accuracy_score,
                "precision": skmetrics.precision_score,
                "recall": skmetrics.recall_score,
            }
        elif self.sub_task == "multi_class_classification":
            self.valid_metrics = {
                "logloss": partial(skmetrics.log_loss, labels=self.labels),
                "accuracy": skmetrics.accuracy_score,
                "mlogloss": partial(skmetrics.log_loss, labels=self.labels),
                "f1_macro": partial(skmetrics.f1_score, average="macro", labels=self.labels),
                "f1_micro": partial(skmetrics.f1_score, average="micro", labels=self.labels),
                "f1_weighted": partial(skmetrics.f1_score, average="weighted", labels=self.labels),
                "precision_macro": partial(skmetrics.precision_score, average="macro", labels=self.labels),
                "precision_micro": partial(skmetrics.precision_score, average="micro", labels=self.labels),
                "precision_weighted": partial(skmetrics.precision_score, average="weighted", labels=self.labels),
                "recall_macro": partial(skmetrics.recall_score, average="macro", labels=self.labels),
                "recall_micro": partial(skmetrics.recall_score, average="micro", labels=self.labels),
                "recall_weighted": partial(skmetrics.recall_score, average="weighted", labels=self.labels),
            }
        elif self.sub_task in ("single_column_regression", "multi_column_regression"):
            self.valid_metrics = {
                "r2": skmetrics.r2_score,
                "mse": skmetrics.mean_squared_error,
                "mae": skmetrics.mean_absolute_error,
                "rmse": partial(skmetrics.mean_squared_error, squared=False),
                "rmsle": partial(skmetrics.mean_squared_log_error, squared=False),
            }
        elif self.sub_task == "multi_label_classification":
            self.valid_metrics = {
                "logloss": skmetrics.log_loss,
            }
        else:
            raise ValueError("Invalid problem type")

    def calculate(self, y_true, y_pred):
        metrics = {}
        for metric_name, metric_func in self.valid_metrics.items():
            if self.sub_task == "binary_classification":
                if metric_name == "auc":
                    metrics[metric_name] = metric_func(y_true, y_pred[:, 1])
                elif metric_name == "logloss":
                    metrics[metric_name] = metric_func(y_true, y_pred)
                else:
                    metrics[metric_name] = metric_func(y_true, y_pred[:, 1] >= 0.5)
            elif self.sub_task == "multi_class_classification":
                if metric_name in (
                    "accuracy",
                    "f1_macro",
                    "f1_micro",
                    "f1_weighted",
                    "precision_macro",
                    "precision_micro",
                    "precision_weighted",
                    "recall_macro",
                    "recall_micro",
                    "recall_weighted",
                ):
                    metrics[metric_name] = metric_func(y_true, np.argmax(y_pred, axis=1))
                else:
                    metrics[metric_name] = metric_func(y_true, y_pred)
            else:
                if metric_name == "rmsle":
                    temp_pred = copy.deepcopy(y_pred)
                    temp_pred = np.clip(temp_pred, 0, None)
                    metrics[metric_name] = metric_func(y_true, temp_pred)
                else:
                    metrics[metric_name] = metric_func(y_true, y_pred)
        return metrics


class TabularModel:
    def __init__(self, model, preprocessor, sub_task, params):
        self.model = model
        self.preprocessor = preprocessor
        self.sub_task = sub_task
        self.params = params
        self.use_predict_proba = True

        _model = self._get_model()
        if self.preprocessor is not None:
            self.pipeline = pipeline.Pipeline([("preprocessor", self.preprocessor), ("model", _model)])
        else:
            self.pipeline = pipeline.Pipeline([("model", _model)])

    def _get_model(self):
        if self.model in _MODELS:
            if self.sub_task in CLASSIFICATION_TASKS:
                if self.model in ("svm", "ridge"):
                    self.use_predict_proba = False
                return _MODELS[self.model]["classification"](**self.params)
            elif self.sub_task in REGRESSION_TASKS:
                self.use_predict_proba = False
                return _MODELS[self.model]["regression"](**self.params)
            else:
                raise ValueError("Invalid task")
        else:
            raise ValueError("Invalid model")


def get_params(trial, model, task):
    if model == "xgboost":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.25, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "max_depth": trial.suggest_int("max_depth", 1, 9),
            "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 100, 500),
            "n_estimators": trial.suggest_categorical("n_estimators", [7000, 15000, 20000]),
            "tree_method": "hist",
            "random_state": 42,
        }

        return params

    if model == "logistic_regression":
        if task in CLASSIFICATION_TASKS:
            params = {
                "C": trial.suggest_float("C", 1e-8, 1e3, log=True),
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "n_jobs": -1,
            }
            return params

        raise ValueError("Task not supported")

    if model == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 10000),
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2", None]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "n_jobs": -1,
        }
        if task in CLASSIFICATION_TASKS:
            params["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy"])
            return params
        if task in REGRESSION_TASKS:
            params["criterion"] = trial.suggest_categorical(
                "criterion", ["squared_error", "absolute_error", "poisson"]
            )
            return params
        raise ValueError("Task not supported")

    if model == "extra_trees":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 10000),
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2", None]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "n_jobs": -1,
        }
        if task in CLASSIFICATION_TASKS:
            params["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy"])
            return params
        if task in REGRESSION_TASKS:
            params["criterion"] = trial.suggest_categorical("criterion", ["squared_error", "absolute_error"])
            return params
        raise ValueError("Task not supported")

    if model == "decision_tree":
        params = {
            "max_depth": trial.suggest_int("max_depth", 1, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2", None]),
            "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
        }
        if task in CLASSIFICATION_TASKS:
            params["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy"])
            return params
        if task in REGRESSION_TASKS:
            params["criterion"] = trial.suggest_categorical(
                "criterion", ["squared_error", "absolute_error", "friedman_mse", "poisson"]
            )
            return params
        raise ValueError("Task not supported")

    if model == "linear_regression":
        if task in REGRESSION_TASKS:
            params = {
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            }
            return params
        raise ValueError("Task not supported")

    if model == "svm":
        if task in CLASSIFICATION_TASKS:
            params = {
                "C": trial.suggest_float("C", 1e-8, 1e3, log=True),
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                "penalty": "l2",
                "max_iter": trial.suggest_int("max_iter", 1000, 10000),
            }
            return params

        if task in REGRESSION_TASKS:
            params = {
                "C": trial.suggest_float("C", 1e-8, 1e3, log=True),
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                "loss": trial.suggest_categorical("loss", ["epsilon_insensitive", "squared_epsilon_insensitive"]),
                "epsilon": trial.suggest_float("epsilon", 1e-8, 1e-1, log=True),
                "max_iter": trial.suggest_int("max_iter", 1000, 10000),
            }
            return params
        raise ValueError("Task not supported")

    if model == "ridge":
        params = {
            "alpha": trial.suggest_float("alpha", 1e-8, 1e3, log=True),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "max_iter": trial.suggest_int("max_iter", 1000, 10000),
        }
        if task in CLASSIFICATION_TASKS:
            return params
        if task in REGRESSION_TASKS:
            return params
        raise ValueError("Task not supported")

    if model == "lasso":
        if task in REGRESSION_TASKS:
            params = {
                "alpha": trial.suggest_float("alpha", 1e-8, 1e3, log=True),
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                "max_iter": trial.suggest_int("max_iter", 1000, 10000),
            }
            return params
        raise ValueError("Task not supported")

    if model == "knn":
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 25),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "algorithm": trial.suggest_categorical("algorithm", ["ball_tree", "kd_tree", "brute"]),
            "leaf_size": trial.suggest_int("leaf_size", 1, 100),
            "p": trial.suggest_categorical("p", [1, 2]),
            "metric": trial.suggest_categorical("metric", ["minkowski", "euclidean", "manhattan"]),
        }
        if task in CLASSIFICATION_TASKS or task in REGRESSION_TASKS:
            return params
        raise ValueError("Task not supported")

    return ValueError("Invalid model")


def get_imputer(imputer_name):
    if imputer_name is None:
        return None
    if imputer_name == "median":
        return impute.SimpleImputer(strategy="median")
    if imputer_name == "mean":
        return impute.SimpleImputer(strategy="mean")
    if imputer_name == "most_frequent":
        return impute.SimpleImputer(strategy="most_frequent")
    raise ValueError("Invalid imputer")


def get_scaler(scaler_name):
    if scaler_name is None:
        return None
    if scaler_name == "standard":
        return preprocessing.StandardScaler()
    if scaler_name == "minmax":
        return preprocessing.MinMaxScaler()
    if scaler_name == "robust":
        return preprocessing.RobustScaler()
    if scaler_name == "normal":
        return preprocessing.Normalizer()
    raise ValueError("Invalid scaler")


def get_metric_direction(sub_task):
    if sub_task == "binary_classification":
        return "logloss", "minimize"
    if sub_task == "multi_class_classification":
        return "mlogloss", "minimize"
    if sub_task == "single_column_regression":
        return "rmse", "minimize"
    if sub_task == "multi_label_classification":
        return "logloss", "minimize"
    if sub_task == "multi_column_regression":
        return "rmse", "minimize"
    raise ValueError("Invalid sub_task")


def get_categorical_columns(df):
    return list(df.select_dtypes(include=["category", "object"]).columns)


def get_numerical_columns(df):
    return list(df.select_dtypes(include=["number"]).columns)


def create_model_card(config, sub_task, best_params, best_metrics):
    best_metrics = "\n".join([f"- {k}: {v}" for k, v in best_metrics.items()])
    best_params = "\n".join([f"- {k}: {v}" for k, v in best_params.items()])
    return MARKDOWN.format(
        task=config.task,
        dataset=config.data_path,
        metrics=best_metrics,
        params=best_params,
    )
