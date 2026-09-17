"""Train multinomial logistic regression; nothing runs on import."""

from typing import Any

import pandas as pd
from numpy.typing import NDArray

from Models.custom_model import custom_model

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = [
    "country/region",
    "previous_majority_proportion",
    "previous_winner",
    "Conservative",
    "Labour",
    "LD",
    "incumbent",
    "previous_con_share",
    "previous_lib_share",
    "previous_lab_share",
    "previous_natSW_share",
    "projected_con_share",
    "projected_lib_share",
    "projected_lab_share",
]
CATEGORICAL_COLUMNS = ["country/region", "previous_winner", "incumbent"]
NUMERIC_COLUMNS = [
    column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS
]


def filter_logistic_regression_data(data: pd.DataFrame) -> pd.DataFrame:
    """Keep complete rows without 'oth' in winner or previous_winner."""
    # Modelling rationale: 'oth' observations are highly influential, so remove
    # rows where either the current winner or previous winner is 'oth'.
    eligible = data.loc[~data[["winner", "previous_winner"]].eq("oth").any(axis=1)]
    return eligible.dropna(subset=FEATURE_COLUMNS + ["winner"]).copy()


def train_logistic_regression(data: pd.DataFrame) -> Pipeline:
    """Return a fitted Pipeline excluding 'oth' outcomes and previous winners.

    Pool all supplied elections, excluding election itself as a predictor.
    Drop rows with 'oth' in winner or previous_winner, or missing a predictor
    or winner, without imputing values.
    The numeric predictors are considered approximately linear in the log odds,
    so no major issues with this assumption are expected. This assumption does
    not apply to categorical predictors, which are one-hot encoded.
    Numeric predictors are scaled to help convergence. Prediction data must
    also have complete predictors. The input dataframe is not modified.
    """
    training = filter_logistic_regression_data(data)
    preprocessing = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ("numeric", StandardScaler(), NUMERIC_COLUMNS),
    ])
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(solver="lbfgs", max_iter=1000)),
    ])
    model.fit(training[FEATURE_COLUMNS], training["winner"])
    return model


class LogisticRegressionModel(custom_model):
    """Own the fitted pipeline; retraining repeats the existing training procedure."""

    def __init__(self) -> None:
        super().__init__("Logistic Regression")
        self.pipeline: Pipeline | None = None

    def train(self, data: pd.DataFrame) -> None:
        """Replace the fitted pipeline using this model's original training routine."""
        self.pipeline = train_logistic_regression(data)

    def predict(self, data: pd.DataFrame) -> NDArray[Any]:
        """Select this model's predictors and return the original party labels."""
        if self.pipeline is None:
            raise RuntimeError("Call train() before predict().")
        return self.pipeline.predict(data[FEATURE_COLUMNS])
