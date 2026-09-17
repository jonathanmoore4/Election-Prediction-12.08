"""Select using 2019 validation results, then retrain on all supplied data."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import accuracy_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Models import NN01_model, logistic_regression, random_forest, xgboost_model
from Models.custom_model import custom_model


def automated_model_selection(data: pd.DataFrame) -> list[custom_model | str]:
    """Train before 2019, validate on 2019, and refit the best model on data.

    Accept numeric or string election years. Evaluate all models on the same
    complete validation rows because logistic regression does not impute.
    Retain all party classes. Ties favour XGBoost, then random forest, then
    logistic regression, then neural network. Each trainer applies its own
    missing-data handling.

    The winning object retrains on the entire dataframe, including 2019 and
    any later years, using its own retraining procedure.
    Return [fitted_model, model_name], with model_name a readable string,
    without reading files or modifying input data.
    """
    feature_columns = logistic_regression.FEATURE_COLUMNS
    required_columns = ["election", "winner"] + feature_columns
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Data is missing required columns: {missing_columns}")

    election_years = pd.to_numeric(data["election"], errors="raise")
    if election_years.isna().any():
        raise ValueError("Election years must not be missing.")

    training_data = data.loc[election_years < 2019].copy()
    validation_data = data.loc[election_years == 2019].dropna(
        subset=feature_columns + ["winner"]
    )
    if training_data.empty:
        raise ValueError("Data must contain training rows from elections before 2019.")
    if validation_data.empty:
        raise ValueError("Data must contain complete validation rows for the 2019 election.")

    models: list[custom_model] = [
        xgboost_model.XGBoostModel(),
        random_forest.RandomForestModel(),
        logistic_regression.LogisticRegressionModel(),
        NN01_model.NeuralNetworkModel(),
    ]
    best_model: custom_model | None = None
    best_accuracy = -1.0

    for model in models:
        model.train(training_data)
        predictions = model.predict(validation_data)
        accuracy = accuracy_score(validation_data["winner"], predictions)
        if accuracy > best_accuracy:
            best_model = model
            best_accuracy = accuracy

    # Strict comparison above preserves the existing ordering for tied scores.
    assert best_model is not None
    best_model.retrain(data)
    return [best_model, best_model.name]
