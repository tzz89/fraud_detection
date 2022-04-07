"""
Implements trainer modules for model selection
"""
import os
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import ParameterGrid, cross_validate
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from lightgbm import LGBMClassifier
from dotenv import load_dotenv
import numpy as np
import joblib
import pandas as pd
import wandb
from .utils import load_yaml



def create_pipeline(model_name: str, model_params: dict) -> Pipeline:
    """
    Creates pipeline based on model_name and model_params
    Args:
        model_name (str): name of model
        model_param (dict): dictionary of model params
    Returns:
        Pipeline: constructed pipeline
    """
    scaler = RobustScaler()
    model_map = {
        "RandomForestClassifier": RandomForestClassifier, "IsolationForest": IsolationForest,
        "LogisticRegression": LogisticRegression, "LinearSVC": LinearSVC,
        "LGBMClassifier": LGBMClassifier}

    model = model_map[model_name](**model_params)
    pipeline = Pipeline(steps=[("scaler", scaler), ("model", model)])

    return pipeline


def calulate_metrics(y_true, y_proba, metrics_list, is_test):
    """
    Calculates the binary metrics
    """
    if len(y_proba.shape) > 1:
        y_pred = np.argmax(y_proba, axis=1)
        y_score = y_proba[:, 1]
    else:
        y_pred = y_proba > 0.5
        y_score = y_proba

    metric_result = {}

    for metric_name in metrics_list:
        if metric_name == "average_precision":
            metric_result["average_precision"] = average_precision_score(
                y_true, y_score
            )
        elif metric_name == "f1":
            metric_result["f1"] = f1_score(y_true, y_pred)
        elif metric_name == "f1_micro":
            metric_result["f1_micro"] = f1_score(y_true, y_pred, average="micro")
        elif metric_name == "f1_macro":
            metric_result["f1_macro"] = f1_score(y_true, y_pred, average="macro")
        elif metric_name == "precision":
            metric_result["precision"] = precision_score(y_true, y_pred)
        elif metric_name == "recall":
            metric_result["recall"] = recall_score(y_true, y_pred)
        elif metric_name == "roc_auc":
            metric_result["roc_auc"] = roc_auc_score(y_true, y_score)
        elif metric_name == "accuracy":
            metric_result["accuracy"] = accuracy_score(y_true, y_pred)

    if is_test:
        metric_result = {"test_" + k: v for k, v in metric_result.items()}
    else:
        metric_result = {"train_" + k: v for k, v in metric_result.items()}

    return metric_result


def trainer(
    config_fp="configs/main_config.yaml",
    train_data_fp="data/train.csv",
    golden_data_fp="data/golden.csv",
    saved_model_dir="models",
    metadata_dir="metadata",
):
    """This function will perform gridsearch training on the various models
       specified in the configuration file

    Args:
        config_fp (str, optional): _description_. Defaults to "configs/main_config.yaml".
        train_data_fp (str, optional): _description_. Defaults to "data/train.csv".
        golden_data_fp (str, optional): _description_. Defaults to "data/golden.csv".
        saved_model_dir (str, optional): _description_. Defaults to "models".
        metadata_dir (str, optional): _description_. Defaults to "metadata".

    Returns:
        _type_: _description_
    """

    # loading config and sercrets
    config = load_yaml(config_fp)
    load_dotenv()

    # loading in training data and golden data
    train_df = pd.read_csv(train_data_fp)
    X = train_df.drop("Class", axis=1)
    y = train_df["Class"]

    golden_df = pd.read_csv(golden_data_fp)
    X_test = golden_df.drop("Class", axis=1)
    y_test = golden_df["Class"]

    best_model_metrics = {}

    for model_name in config["models"]:
        print(f"\nPerforming training on model {model_name}")
        best_model_params = None
        best_optimizing_metric_score = -1

        for model_params in ParameterGrid(config["models"][model_name]):
            # start new run
            wandb.init(
                project=config["wandb_params"]["project_name"],
                entity=config["wandb_params"]["entity"],
            )
            print(f"Params {model_params}")
            pipeline = create_pipeline(model_name, model_params)

            # 5 fold cross validation
            cv_scores = cross_validate(
                pipeline,
                X,
                y,
                scoring=config["metrics"]["tracking_metrics"],
                cv=config["training_params"]["n_folds"],
                n_jobs=config["training_params"]["n_jobs"],
                return_train_score=True,
            )

            average_cv_scores = {k: np.mean(v) for k, v in cv_scores.items()}

            # save best_model
            optimizing_metric_score = average_cv_scores[
                config["metrics"]["optimizing_metric"]
            ]
            if (
                best_optimizing_metric_score == -1
                or optimizing_metric_score > best_optimizing_metric_score
            ):
                best_optimizing_metric_score = optimizing_metric_score
                best_model_params = model_params

            # logging to weights and biass
            wandb_cv_logs = {
                "model_name": model_name,
                "on_golden_test": False,
                **model_params,
                **average_cv_scores,
            }
            wandb.log(wandb_cv_logs)

        # train best params with full training data
        best_pipeline = create_pipeline(model_name, best_model_params)
        best_pipeline.fit(X, y)

        # saving model
        joblib.dump(
            best_pipeline, os.path.join(saved_model_dir, f"{model_name}_best.pkl")
        )

        # test best_model on golden test set
        try:
            train_y_proba = best_pipeline.predict_proba(X)
            test_y_proba = best_pipeline.predict_proba(X_test)
        except AttributeError as exception:
            print(exception)
            train_y_proba = best_pipeline.predict(X)
            test_y_proba = best_pipeline.predict(X_test)

        # calculates training and test scores
        train_metrics = calulate_metrics(
            y, train_y_proba, config["metrics"]["tracking_metrics"], is_test=False
        )
        test_metrics = calulate_metrics(
            y_test, test_y_proba, config["metrics"]["tracking_metrics"], is_test=True
        )
        wandb_best_logs = {
            "model_name": model_name,
            "on_golden_test": True,
            **best_model_params,
            **train_metrics,
            **test_metrics,
        }

        wandb.log(wandb_best_logs)
        # save best_model metrics
        best_model_metrics[model_name] = {**train_metrics, **test_metrics}

        # wandb.sklearn.plot_classifier
        if len(test_y_proba.shape) > 1:
            y_pred = np.argmax(test_y_proba, axis=1)

        else:
            y_pred = test_y_proba > 0.5
            test_y_proba = np.hstack(
                [1 - test_y_proba.reshape(-1, 1), test_y_proba.reshape(-1, 1)]
            )

        wandb.sklearn.plot_classifier(
            best_pipeline,
            X,
            X_test,
            y,
            y_test,
            y_pred,
            test_y_proba,
            labels=["not fraud", "fraud"],
            model_name=model_name,
        )
    wandb.finish()

    # saving model metrics to metadata
    best_model_metrics_df = pd.DataFrame(best_model_metrics)
    best_model_metrics_df.to_csv(os.path.join(metadata_dir, "best_model_metrics.csv"))
    return best_model_metrics_df
