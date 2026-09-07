"""Select using 2019 validation results, then retrain on all supplied data."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import accuracy_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Models import logistic_regression, random_forest, xgboost_model


def automated_model_selection(data):
    """Train before 2019, validate on 2019, and refit the best model on data.

    Accept numeric or string election years. Evaluate all models on the same
    complete validation rows because logistic regression does not impute.
    Retain all party classes. Ties favour XGBoost, then random forest, then
    logistic regression. Each trainer applies its own missing-data handling.

    The winning training function is called again with the entire dataframe,
    including 2019 and any later years. For XGBoost this repeats grid search.
    Return the fitted pipeline without reading files or modifying input data.
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

    model_functions = [
        xgboost_model.train_xgboost,
        random_forest.train_random_forest,
        logistic_regression.train_logistic_regression,
    ]
    best_training_function = None
    best_accuracy = -1.0

    for train_model in model_functions:
        model = train_model(training_data)
        predictions = model.predict(validation_data[feature_columns])
        accuracy = accuracy_score(validation_data["winner"], predictions)
        if accuracy > best_accuracy:
            best_training_function = train_model
            best_accuracy = accuracy

    return best_training_function(data)
