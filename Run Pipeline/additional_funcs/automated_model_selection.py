"""Select using 2019 validation results, then retrain using each model's procedure."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import accuracy_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Models import NN01_model, NN02_model, NN03_model, logistic_regression, random_forest, xgboost_model
from Models import xgboost_expanded_model
from Models.custom_model import custom_model


def automated_model_selection(
    data: pd.DataFrame, scores_path: Path | None = None,
) -> list[custom_model | str]:
    """Train before 2019, validate on 2019, and refit the best model on data.

    Accept numeric or string election years. Evaluate all models on the same
    complete validation rows because logistic regression does not impute.
    Retain all party classes. Ties favour XGBoost, then random forest, then
    logistic regression, then NN01, NN02, NN03 and XGBoost Expanded in that order. Each trainer applies its own
    missing-data handling.

    The winning object receives the entire training dataframe for retraining.
    Each neural network holds out its latest whole election for early stopping
    across ten seeds. NN01 searches four rates with 32/16 layers; NN02 searches
    eight rates with 64/32 layers; NN03 uses one 16-unit layer and fixed rate 0.3.
    Keep the prediction election (2024) outside this dataframe.
    Return [fitted_model, model_name], with model_name a readable string,
    without reading files or modifying input data. Print all initial accuracies
    and retain them on fitted_model.initial_accuracies. If scores_path is supplied,
    save the scores there before retraining. Also record accuracy on complete
    2019 rows where winner differs from a known previous_winner. Select by the
    sum of overall and changed-seat accuracy, weighting both equally. An empty
    changed-seat subset has accuracy None; selection then uses overall accuracy.
    """
    feature_columns = logistic_regression.FEATURE_COLUMNS
    required_columns = list(dict.fromkeys(
        ["election", "winner", "previous_winner"]
        + feature_columns + xgboost_expanded_model.FEATURE_COLUMNS
    ))
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

    changed_seats = (
        validation_data["previous_winner"].notna()
        & validation_data["winner"].ne(validation_data["previous_winner"])
    ).to_numpy(dtype=bool)
    changed_seat_count = int(changed_seats.sum())

    models: list[custom_model] = [
        xgboost_model.XGBoostModel(),
        random_forest.RandomForestModel(),
        logistic_regression.LogisticRegressionModel(),
        NN01_model.NeuralNetworkModel(),
        NN02_model.NeuralNetworkModel(),
        NN03_model.NeuralNetworkModel(),
        xgboost_expanded_model.XGBoostExpandedModel(),
    ]
    best_model: custom_model | None = None
    best_selection_score = -1.0
    initial_accuracies: dict[str, float] = {}
    changed_seat_accuracies: dict[str, float | None] = {}

    print(f"Initial model accuracies on {len(validation_data)} complete 2019 rows:", flush=True)
    for model in models:
        model.train(training_data)
        predictions = model.predict(validation_data)
        accuracy = float(accuracy_score(validation_data["winner"], predictions))
        initial_accuracies[model.name] = accuracy
        changed_accuracy = (
            float(accuracy_score(
                validation_data.loc[changed_seats, "winner"],
                pd.Series(predictions).to_numpy()[changed_seats],
            ))
            if changed_seat_count else None
        )
        changed_seat_accuracies[model.name] = changed_accuracy
        selection_score = accuracy + (changed_accuracy if changed_accuracy is not None else 0.0)
        changed_display = f"{changed_accuracy:.2%}" if changed_accuracy is not None else "N/A"
        print(
            f"  {model.name}: {accuracy:.2%}; changed winning party: "
            f"{changed_display} ({changed_seat_count} seats); "
            f"combined score: {selection_score:.4f}", flush=True,
        )
        if selection_score > best_selection_score:
            best_model = model
            best_selection_score = selection_score

    # Strict comparison above preserves the existing ordering for tied scores.
    assert best_model is not None
    best_model.initial_accuracies = initial_accuracies
    best_model.initial_changed_seat_accuracies = changed_seat_accuracies
    best_model.changed_seat_evaluation_rows = changed_seat_count
    if scores_path is not None:
        pd.DataFrame([
            {"model": name, "accuracy": score, "evaluation_year": 2019,
             "evaluation_rows": len(validation_data),
             "changed_seat_accuracy": changed_seat_accuracies[name],
             "changed_seat_evaluation_rows": changed_seat_count,
             "selection_score": score + (
                 changed_seat_accuracies[name]
                 if changed_seat_accuracies[name] is not None else 0.0
             )}
            for name, score in initial_accuracies.items()
        ]).to_csv(scores_path, index=False)
    print(f"Selected {best_model.name}; retraining using its model-specific procedure.", flush=True)
    best_model.retrain(data)
    return [best_model, best_model.name]
