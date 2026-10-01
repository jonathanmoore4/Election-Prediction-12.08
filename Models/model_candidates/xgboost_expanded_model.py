"""XGBoost Expanded: one configuration per fit; tuning is owned by the evaluator."""
from Models.pipeline_model import PipelineModel
from Models.boosting import BOOST_DEFAULTS, fit_boosting_pipeline

FEATURE_COLUMNS = [
    "country/region",
    "previous_con_national_vote_share",
    "previous_lab_national_vote_share",
    "previous_lib_national_vote_share",
    "previous_natSW_national_vote_share",
    "previous_oth_national_vote_share",
    "previous_winning_party_last_election_vote_share",
    "previous_second_party_last_election_vote_share",
    "previous_winner",
    "previous_con_share",
    "previous_lib_share",
    "previous_lab_share",
    "previous_natSW_share",
    "con_polling",
    "lab_polling",
    "lib_polling",
    "incumbent",
    "supported_incumbent",
    "incumbent_polling",
    "opposition_polling",
    "con_national_change",
    "lab_national_change",
    "lib_national_change",
    "natSW_national_change",
    "oth_national_change",
    "previous_margin_1st_2nd",
    "projected_con_share",
    "projected_lib_share",
    "projected_lab_share",
    "holder_national_swing",
    "challenger_national_swing",
    "holder_polling",
    "challenger_polling",
]
CATEGORICAL_COLUMNS = ["country/region", "previous_winner", "incumbent"]
NUMERIC_COLUMNS = [
    column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS
]



def train_xgboost_expanded(data, configuration=None):
    return fit_boosting_pipeline(data, FEATURE_COLUMNS, CATEGORICAL_COLUMNS,
                                 {**BOOST_DEFAULTS, **(configuration or {})})


class XGBoostExpandedModel(PipelineModel):
    feature_columns = FEATURE_COLUMNS
    defaults = BOOST_DEFAULTS

    def __init__(self):
        super().__init__('XGBoost Expanded')

    def fit_pipeline(self, data, parameters):
        return train_xgboost_expanded(data, parameters)
