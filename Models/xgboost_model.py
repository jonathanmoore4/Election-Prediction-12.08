"""Tune XGBoost with expanding election cross-validation; nothing runs on import."""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from Models.custom_model import custom_model

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

FEATURE_COLUMNS = [
    "country/region",
    "previous_winning_party_last_election_vote_share",
    "previous_winner",
    "con_polling",
    "lab_polling",
    "lib_polling",
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


class _LabelledXGBClassifier(ClassifierMixin, BaseEstimator):
    """Keep XGBoost's integer target encoding inside a sklearn estimator."""

    def __init__(self, n_estimators=100, max_depth=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()
        encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            objective="multi:softprob",
            num_class=len(self.classes_),
            eval_metric="mlogloss",
            n_jobs=1,
            random_state=42,
        )
        self.model_.fit(X, encoded)
        self.n_features_in_ = self.model_.n_features_in_
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        encoded = self.model_.predict(X).astype(int)
        return self.label_encoder_.inverse_transform(encoded)

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(X)


def train_xgboost(data: pd.DataFrame) -> Pipeline:
    """Return the best fitted Pipeline from the notebook's accuracy grid search.

    Retain all winner classes, including 'oth'. Election is metadata, not a
    predictor. Rows without a winner are dropped; XGBoost handles missing
    numeric predictors. Expanding-window CV validates on each supplied election
    after 2001, training on all earlier elections. Mean accuracy weights each
    validation election equally. The winner is refitted on all labelled rows.
    Callers must exclude any held-out elections from data. Preprocessing and
    target encoding are fitted inside each fold. Predictions return party labels, and
    predict_proba columns follow classes_. The input dataframe is not modified.
    """
    training = data.dropna(subset=["winner"])
    # These positions index the same labelled rows supplied to search.fit.
    cv_years = pd.to_numeric(training["election"], errors="raise").to_numpy()
    if not np.isfinite(cv_years).all():
        raise ValueError("Every training row must have a valid election year.")
    validation_years = sorted(year for year in np.unique(cv_years) if year > 2001)
    election_splits = []
    for validation_year in validation_years:
        train_indices = np.flatnonzero(cv_years < validation_year)
        validation_indices = np.flatnonzero(cv_years == validation_year)
        if not len(train_indices):
            raise ValueError(f"No earlier training rows for election {validation_year}.")
        election_splits.append((train_indices, validation_indices))
    if len(election_splits) < 2:
        raise ValueError("At least two validation elections after 2001 are required.")

    preprocessing = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_COLUMNS),
        ("numeric", "passthrough", NUMERIC_COLUMNS),
    ])
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", _LabelledXGBClassifier()),
    ])
    search = GridSearchCV(
        estimator=model,
        param_grid={
            "classifier__n_estimators": [20, 35, 50, 100],
            "classifier__max_depth": [2, 3, 4, 5],
        },
        scoring="accuracy",
        cv=election_splits,
        refit=True,
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(training[FEATURE_COLUMNS], training["winner"])
    return search.best_estimator_


class XGBoostModel(custom_model):
    """Own the fitted pipeline; retraining repeats the existing training procedure."""

    def __init__(self) -> None:
        super().__init__("XGBoost")
        self.pipeline: Pipeline | None = None

    def train(self, data: pd.DataFrame) -> None:
        """Replace the fitted pipeline using this model's original training routine."""
        self.pipeline = train_xgboost(data)

    def predict(self, data: pd.DataFrame) -> NDArray[Any]:
        """Select this model's predictors and return the original party labels."""
        if self.pipeline is None:
            raise RuntimeError("Call train() before predict().")
        return self.pipeline.predict(data[FEATURE_COLUMNS])
