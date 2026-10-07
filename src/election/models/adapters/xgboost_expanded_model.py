"""XGBoost Expanded: one configuration per fit; tuning is owned by the evaluator."""
import election.models.custom_model as custom_model_module
import election.models.pipeline_model as pipeline_model_module
import election.models.boosting as boosting_module

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
    return boosting_module.fit_boosting_pipeline(data, FEATURE_COLUMNS, CATEGORICAL_COLUMNS,
                                 {**boosting_module.BOOST_DEFAULTS, **(configuration or {})})


class XGBoostExpandedModel(pipeline_model_module.PipelineModel):
    feature_columns = FEATURE_COLUMNS
    defaults = boosting_module.BOOST_DEFAULTS

    def __init__(self):
        super().__init__('XGBoost Expanded')

    def fit_pipeline(self, data, parameters):
        return train_xgboost_expanded(data, parameters)


xgboostexpandedhyperparameters = {'n_estimators': [20, 35, 50, 100], 'max_depth': [2, 3, 4, 5]}
TRAINING_METADATA = {
    'model_id': 'xgboost_expanded',
    'architecture_id': 'xgboost_expanded_model-v1',
    'training_protocol': 'full-history-v1',
    'fixed_settings': dict(boosting_module.BOOST_DEFAULTS),
    'features': list(FEATURE_COLUMNS),
    'supported_hyperparameters': sorted(set(boosting_module.BOOST_DEFAULTS) | set(xgboostexpandedhyperparameters)),
}


def XGBoostExpanded(train_data, test_data, hyperparameters, *, fit_records=None,
           cache=None, return_details=False, output_dir=None, return_model=False):
    """Fit one configuration and score known winners; no MLflow operations."""
    from election.models.model_function import evaluate_fit
    return evaluate_fit(XGBoostExpandedModel, train_data, test_data, hyperparameters,
                        metadata=TRAINING_METADATA,
                        context=custom_model_module.fit_context(cache=cache),
                        return_details=return_details, output_dir=output_dir, return_model=return_model)
