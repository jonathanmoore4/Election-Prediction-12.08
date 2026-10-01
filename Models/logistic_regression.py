"""Train multinomial logistic regression; nothing runs on import."""

import numpy as np
import pandas as pd

from Models.pipeline_model import PipelineModel
from Models.missing_data import ElectionImputer

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

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


def normalise_predictors(data: pd.DataFrame) -> pd.DataFrame:
    """Normalise SQL/CSV dtypes before fitting or applying missing-value fills."""
    result = data.copy()
    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(result[column], errors="raise").astype(float)
        result[column] = values.replace([np.inf, -np.inf], np.nan)
    for column in CATEGORICAL_COLUMNS:
        values = result[column].astype(object)
        result[column] = values.where(values.notna(), np.nan)
    return result


def filter_logistic_regression_data(data: pd.DataFrame) -> pd.DataFrame:
    """Legacy notebook helper; the shared evaluator does not apply this filter."""
    # Modelling rationale: 'oth' observations are highly influential, so remove
    # rows where either the current winner or previous winner is 'oth'.
    eligible = data.loc[~data[["winner", "previous_winner"]].eq("oth").any(axis=1)]
    return eligible.dropna(subset=["winner"]).copy()


def train_logistic_regression(data: pd.DataFrame, configuration=None) -> Pipeline:
    """Fit one configuration on all labelled rows, including the oth class."""
    training = data.dropna(subset=['winner']).copy()
    parameters = {**LOGISTIC_DEFAULTS, **(configuration or {})}
    estimator_parameters = parameters.copy()
    api_parameters = LogisticRegression().get_params()
    # sklearn >= 1.8 expresses L2 through l1_ratio; older versions use penalty.
    if estimator_parameters.get('penalty') == 'l2' and (
            'penalty' not in api_parameters or api_parameters['penalty'] == 'deprecated'):
        estimator_parameters.pop('penalty')
        estimator_parameters['l1_ratio'] = 0.0
    preprocessing = ColumnTransformer([
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_COLUMNS),
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), NUMERIC_COLUMNS),
    ])
    model = Pipeline([
        ("normalise", FunctionTransformer(normalise_predictors, validate=False)),
        ("missing_data", ElectionImputer()),
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(**estimator_parameters)),
    ])
    # A one-class history is a declared constant-probability fallback.
    if training.winner.nunique() == 1:
        from sklearn.dummy import DummyClassifier
        model.set_params(classifier=DummyClassifier(strategy='prior'))
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(training[FEATURE_COLUMNS + ["election"]], training["winner"])
    return model


LOGISTIC_DEFAULTS = dict(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000,
                         tol=0.0001, class_weight=None)


class LogisticRegressionModel(PipelineModel):
    feature_columns = FEATURE_COLUMNS
    defaults = LOGISTIC_DEFAULTS

    def __init__(self):
        super().__init__('Logistic Regression')

    def fit_pipeline(self, data, parameters):
        return train_logistic_regression(data, parameters)
