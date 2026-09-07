"""Tune XGBoost on pooled election data; nothing runs on import."""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

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


def train_xgboost(data):
    """Return the best fitted Pipeline from the notebook's accuracy grid search.

    Pool all supplied elections and retain all winner classes, including 'oth'.
    Election is metadata, not a predictor. Rows without a winner are dropped;
    XGBoost handles missing numeric predictors. Five-fold stratified CV selects
    parameters, then refits the winner on all labelled rows. Preprocessing is
    fitted inside each fold. Predictions return original party labels, and
    predict_proba columns follow classes_. The input dataframe is not modified.
    """
    training = data.dropna(subset=["winner"])
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
        cv=5,
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(training[FEATURE_COLUMNS], training["winner"])
    return search.best_estimator_

