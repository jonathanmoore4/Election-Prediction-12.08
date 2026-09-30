"""Select using 2019 validation results, then retrain using each model's procedure."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import accuracy_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Models import NN01_model, NN02_model, NN03_model, logistic_regression, random_forest, xgboost_model
from Models import xgboost_expanded_model, conditional_xgboost_model
from Models.custom_model import custom_model


def automated_model_selection(
    data: pd.DataFrame, scores_path: Path | None = None,
) -> list[custom_model | str]:
    """Train before 2019, validate on 2019, and refit the best model on data.

    Accept numeric or string election years. Evaluate all models on the same
    labelled validation rows, with training-fitted imputation in each model.
    Retain all party classes. Ties favour XGBoost, then random forest, then
    logistic regression, then NN01, NN02, NN03, XGBoost Expanded and Conditional XGBoost in that order. Each trainer applies its own
    missing-data handling.

    The winning object receives the entire training dataframe for retraining.
    Each neural network holds out its latest whole election for early stopping
    across ten seeds. NN01 searches four rates with 32/16 layers; NN02 searches
    eight rates with 64/32 layers; NN03 uses one 16-unit layer and fixed rate 0.3.
    Keep the prediction election (2024) outside this dataframe.
    Return [fitted_model, model_name], with model_name a readable string,
    without reading files or modifying input data. Print all initial accuracies
    and retain them on fitted_model.initial_accuracies. If scores_path is supplied,
    save the scores there before retraining. Also record accuracy on labelled
    2019 rows where winner differs from a known previous_winner. Select by the
    sum of overall and changed-seat accuracy, weighting both equally. An empty
    changed-seat subset has accuracy None; selection then uses overall accuracy.
    """
    feature_columns = logistic_regression.FEATURE_COLUMNS
    required_columns = list(dict.fromkeys(
        ["election", "winner", "previous_winner"]
        + feature_columns + xgboost_expanded_model.FEATURE_COLUMNS
        + conditional_xgboost_model.FEATURE_COLUMNS
    ))
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Data is missing required columns: {missing_columns}")

    election_years = pd.to_numeric(data["election"], errors="raise")
    if election_years.isna().any():
        raise ValueError("Election years must not be missing.")

    training_data = data.loc[election_years < 2019].copy()
    validation_data = data.loc[election_years == 2019].dropna(
        subset=["winner"]
    )
    if training_data.empty:
        raise ValueError("Data must contain training rows from elections before 2019.")
    if validation_data.empty:
        raise ValueError("Data must contain labelled validation rows for the 2019 election.")

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
        conditional_xgboost_model.ConditionalXGBoostModel(),
    ]
    best_model: custom_model | None = None
    best_selection_score = -1.0
    initial_accuracies: dict[str, float] = {}
    changed_seat_accuracies: dict[str, float | None] = {}

    # Report the shared validation sample size; flush=True prints progress immediately.
    print(f"Initial model accuracies on {len(validation_data)} labelled 2019 rows:", flush=True)
    # Evaluate candidates in list order, which also sets the priority for tied scores.
    for model in models:
        # Fit this candidate on elections before 2019 using its own training procedure.
        model.train(training_data)
        # Produce one predicted winning-party label for each 2019 validation row.
        predictions = model.predict(validation_data)
        # Compare predictions with actual winners; accuracy is a fraction from 0 to 1.
        accuracy = float(accuracy_score(validation_data["winner"], predictions))
        # Keep this candidate's overall validation accuracy under its readable name.
        initial_accuracies[model.name] = accuracy
        # Also score only seats whose winner differs from a known previous winner.
        changed_accuracy = (
            float(accuracy_score(
                # Select actual winners using the Boolean mask built above.
                validation_data.loc[changed_seats, "winner"],
                # Convert predictions to an array so the same mask selects by row position.
                pd.Series(predictions).to_numpy()[changed_seats],
            ))
            # Avoid scoring an empty subset; None means the metric is unavailable.
            if changed_seat_count else None
        )
        # Retain the changed-seat result for later reporting and optional CSV output.
        changed_seat_accuracies[model.name] = changed_accuracy
        # Give both accuracies equal weight by adding them (range 0 to 2).
        # If no changed seats exist, use overall accuracy alone (range 0 to 1).
        selection_score = accuracy + (changed_accuracy if changed_accuracy is not None else 0.0)
        # Format the subset accuracy as a percentage with two decimal places, or N/A.
        changed_display = f"{changed_accuracy:.2%}" if changed_accuracy is not None else "N/A"
        print(
            # Adjacent f-strings form one message with overall and subset performance.
            f"  {model.name}: {accuracy:.2%}; changed winning party: "
            f"{changed_display} ({changed_seat_count} seats); "
            # The combined score is a sum, displayed to four decimal places.
            f"combined score: {selection_score:.4f}", flush=True,
        )
        # Replace the leader only for a strictly higher score; ties keep the earlier model.
        if selection_score > best_selection_score:
            # Keep the actual trained object, rather than constructing a new instance.
            best_model = model
            # Update the threshold against which subsequent candidates are compared.
            best_selection_score = selection_score

    # Confirm a candidate was selected; this also narrows its type from optional to model.
    assert best_model is not None
    # Attach all candidates' original validation results to the selected model.
    best_model.initial_accuracies = initial_accuracies
    best_model.initial_changed_seat_accuracies = changed_seat_accuracies
    # Record the number of validation seats used for every changed-seat accuracy.
    best_model.changed_seat_evaluation_rows = changed_seat_count
    # Write a score report only when the caller supplies a destination path.
    if scores_path is not None:
        # Build a table with one dictionary (and therefore one row) per candidate.
        pd.DataFrame([
            # Identify the model, its overall accuracy, and the held-out election year.
            {"model": name, "accuracy": score, "evaluation_year": 2019,
             # Record the sample size behind the overall accuracy.
             "evaluation_rows": len(validation_data),
             # Include the subset metric (None becomes an empty CSV field) and its size.
             "changed_seat_accuracy": changed_seat_accuracies[name],
             "changed_seat_evaluation_rows": changed_seat_count,
             # Recreate the same combined score used to choose the winning model.
             "selection_score": score + (
                 changed_seat_accuracies[name]
                 # Missing subset accuracy contributes zero, as in the selection loop.
                 if changed_seat_accuracies[name] is not None else 0.0
             )}
            # Iterate over every stored model name and overall validation accuracy.
            for name, score in initial_accuracies.items()
        # Save before retraining; omit the DataFrame's generated numeric row index.
        ]).to_csv(scores_path, index=False)
    # Announce the winning candidate before starting its final training stage.
    print(f"Selected {best_model.name}; retraining using its model-specific procedure.", flush=True)
    # Pass the full input dataframe, including 2019, to the winner's retraining method.
    # The caller must exclude the prediction election; each model manages its own holdouts.
    best_model.retrain(data)
    # Return the refitted object and its readable name in the list expected by the caller.
    return [best_model, best_model.name]
