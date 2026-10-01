"""XGBoost: one configuration per fit; tuning is owned by the evaluator."""
from Models.pipeline_model import PipelineModel
from Models.boosting import BOOST_DEFAULTS, fit_boosting_pipeline

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



def train_xgboost(data, configuration=None):
    return fit_boosting_pipeline(data, FEATURE_COLUMNS, CATEGORICAL_COLUMNS,
                                 {**BOOST_DEFAULTS, **(configuration or {})})


class XGBoostModel(PipelineModel):
    feature_columns = FEATURE_COLUMNS
    defaults = BOOST_DEFAULTS

    def __init__(self):
        super().__init__('XGBoost')

    def fit_pipeline(self, data, parameters):
        return train_xgboost(data, parameters)
