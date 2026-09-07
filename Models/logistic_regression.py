"""Train multinomial logistic regression; nothing runs on import."""

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


def train_logistic_regression(data):
    """Return a fitted Pipeline predicting winner labels, including 'oth'.

    Pool all supplied elections, excluding election itself as a predictor.
    Drop rows missing a predictor or winner, without imputing values.
    Numeric predictors are scaled to help convergence. Prediction data must
    also have complete predictors. The input dataframe is not modified.
    """
    training = data.dropna(subset=FEATURE_COLUMNS + ["winner"])
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

