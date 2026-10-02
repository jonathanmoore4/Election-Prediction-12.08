"""Train a random forest on pooled election data; nothing runs on import."""

import pandas as pd

import election.models.pipeline_model as pipeline_model_module
import election.models.missing_data as missing_data_module

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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


def train_random_forest(data: pd.DataFrame, configuration=None) -> Pipeline:
    """Return a fitted Pipeline predicting winner labels, including 'oth'.

    Election is metadata, not a predictor. Rows without a winner are dropped;
    missing numeric predictors are median-imputed. Pass a dataframe containing
    FEATURE_COLUMNS to the returned pipeline's predict or predict_proba method.
    The input dataframe is not modified.
    """
    training = data.dropna(subset=["winner"])
    preprocessing = ColumnTransformer([
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_COLUMNS),
        ("numeric", SimpleImputer(strategy="median"), NUMERIC_COLUMNS),
    ])
    model = Pipeline([
        ("missing_data", missing_data_module.ElectionImputer()),
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(**{**FOREST_DEFAULTS, **(configuration or {})})),
    ])
    model.fit(training[FEATURE_COLUMNS + ["election"]], training["winner"])
    return model


FOREST_DEFAULTS = dict(n_estimators=300, max_depth=None, min_samples_leaf=1,
                       max_features='sqrt', min_samples_split=2, bootstrap=True,
                       criterion='gini', class_weight=None, random_state=42, n_jobs=1)


class RandomForestModel(pipeline_model_module.PipelineModel):
    feature_columns = FEATURE_COLUMNS
    defaults = FOREST_DEFAULTS

    def __init__(self):
        super().__init__('Random Forest')

    def fit_pipeline(self, data, parameters):
        return train_random_forest(data, parameters)
