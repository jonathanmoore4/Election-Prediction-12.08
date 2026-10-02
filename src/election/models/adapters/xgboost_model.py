"""XGBoost: one configuration per fit; tuning is owned by the evaluator."""
import election.models.pipeline_model as pipeline_model_module
import election.models.boosting as boosting_module

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
    return boosting_module.fit_boosting_pipeline(data, FEATURE_COLUMNS, CATEGORICAL_COLUMNS,
                                 {**boosting_module.BOOST_DEFAULTS, **(configuration or {})})


class XGBoostModel(pipeline_model_module.PipelineModel):
    feature_columns = FEATURE_COLUMNS
    defaults = boosting_module.BOOST_DEFAULTS

    def __init__(self):
        super().__init__('XGBoost')

    def fit_pipeline(self, data, parameters):
        return train_xgboost(data, parameters)
