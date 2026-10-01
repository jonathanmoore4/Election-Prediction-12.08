"""The candidate registry is the only list to extend when adding an adapter."""
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable
from sklearn.model_selection import ParameterGrid

from Models.custom_model import custom_model
from Models.training_policy import TrainingPolicy, NeuralTrainingPolicy


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    factory: Callable[[], custom_model]
    search_space: dict = field(default_factory=dict)
    fixed_settings: dict = field(default_factory=dict)
    feature_columns: tuple[str, ...] = ()
    training_policy: TrainingPolicy = field(default_factory=TrainingPolicy)
    version: str = 'nested-election-v1'

    def configurations(self):
        return [{**deepcopy(self.fixed_settings), **deepcopy(c)}
                for c in ParameterGrid(self.search_space)]

    def create_model(self):
        model = self.factory()
        if model.name != self.name:
            raise ValueError(f'Factory name {model.name!r} does not match {self.name!r}.')
        return model


def default_candidates():
    # Lazy imports let tests and lightweight custom registries avoid torch/XGBoost.
    from Models.model_candidates import logistic_regression as lr
    from Models.model_candidates import random_forest as rf
    from Models.model_candidates import xgboost_model as xgb
    from Models.model_candidates import xgboost_expanded_model as expanded
    from Models.model_candidates import conditional_xgboost_model as conditional
    from Models.model_candidates import NN01_model as nn1
    from Models.model_candidates import NN02_model as nn2
    from Models.model_candidates import NN03_model as nn3
    from Models.boosting import BOOST_DEFAULTS
    boost_search = {'n_estimators': [20, 35, 50, 100], 'max_depth': [2, 3, 4, 5]}
    specs = [
        CandidateSpec('XGBoost', xgb.XGBoostModel, boost_search, BOOST_DEFAULTS,
                      tuple(xgb.FEATURE_COLUMNS)),
        CandidateSpec('Random Forest', rf.RandomForestModel,
                      {'max_depth': [5, 10, None], 'min_samples_leaf': [1, 5, 10],
                       'max_features': ['sqrt', 0.5]}, rf.FOREST_DEFAULTS,
                      tuple(rf.FEATURE_COLUMNS)),
        CandidateSpec('Logistic Regression', lr.LogisticRegressionModel,
                      {'C': [0.01, 0.1, 1, 10]}, lr.LOGISTIC_DEFAULTS,
                      tuple(lr.FEATURE_COLUMNS)),
    ]
    for module, name in [(nn1, 'NN01: 32/16 validation-selected rate'),
                         (nn2, 'NN02: 64/32 validation-selected rate'),
                         (nn3, 'NN03: 16 units fixed rate 0.3')]:
        specs.append(CandidateSpec(
            name, module.NeuralNetworkModel, {'learning_rate': list(module.LEARNING_RATES)},
            dict(hidden_sizes=module.HIDDEN_SIZES, seeds=nn1.SEEDS, batch_size=64,
                 max_epochs=1000, patience=20, min_delta=0.0, momentum=0.0, weight_decay=0.0),
            tuple(lr.FEATURE_COLUMNS), NeuralTrainingPolicy()))
    specs.append(CandidateSpec('XGBoost Expanded', expanded.XGBoostExpandedModel,
                               boost_search, BOOST_DEFAULTS, tuple(expanded.FEATURE_COLUMNS)))
    stage_search = {f'{stage}.{key}': values for stage in ('change', 'challenger')
                    for key, values in conditional.PARAM_GRID.items()}
    specs.append(CandidateSpec('Conditional XGBoost', conditional.ConditionalXGBoostModel,
                               stage_search, BOOST_DEFAULTS, tuple(conditional.FEATURE_COLUMNS)))
    return specs
