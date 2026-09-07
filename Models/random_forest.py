"""Train a random forest on pooled election data; nothing runs on import."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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


def train_random_forest(data):
    """Return a fitted Pipeline predicting winner labels, including 'oth'.

    Election is metadata, not a predictor. Rows without a winner are dropped;
    missing numeric predictors are median-imputed. Pass a dataframe containing
    FEATURE_COLUMNS to the returned pipeline's predict or predict_proba method.
    The input dataframe is not modified.
    """
    training = data.dropna(subset=["winner"])
    preprocessing = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ("numeric", SimpleImputer(strategy="median"), NUMERIC_COLUMNS),
    ])
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ])
    model.fit(training[FEATURE_COLUMNS], training["winner"])
    return model

