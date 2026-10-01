"""Small real fits and controlled checkpoint tests for the new adapter contract."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from Models.candidates import default_candidates
from Models.custom_model import FitContext, PARTIES
from Models.evaluation import HistoricalEvaluator
from Models import NN01_model as neural
from Models.training_policy import NeuralTrainingPolicy


def sample():
    specs = default_candidates()
    columns = set(c for s in specs for c in s.feature_columns)
    rows = []
    for year in [1987, 1992, 1997, 2001, 2005, 2010, 2015, 2017, 2019]:
        for i, party in enumerate(PARTIES):
            rows.append({**{c: .1 + .02*i for c in columns}, 'election': year,
                         'winner': party, 'previous_winner': PARTIES[(i+1)%5],
                         'country/region': 'England', 'incumbent': 'con',
                         'constituency_id': f'{year}-{i}'})
    return pd.DataFrame(rows)


def small_spec(spec):
    if isinstance(spec.training_policy, NeuralTrainingPolicy):
        config = {**spec.configurations()[0], 'max_epochs': 2, 'patience': 1,
                  'seeds': (1,), 'learning_rate': .01}
    elif spec.name == 'Conditional XGBoost':
        config = {**spec.configurations()[0], 'change.n_estimators': 2,
                  'challenger.n_estimators': 2}
    else:
        config = spec.configurations()[0]
        if 'n_estimators' in config:
            config['n_estimators'] = 2
    return replace(spec, search_space={}, fixed_settings=config)


def test_registry_search_sizes_match_plan():
    assert [len(s.configurations()) for s in default_candidates()] == [16,18,4,4,8,1,16,256]


@pytest.mark.parametrize('index', range(8))
def test_every_adapter_exposes_aligned_probabilities_and_keeps_rows(index):
    torch.set_num_threads(1)
    spec = small_spec(default_candidates()[index])
    data = sample()
    training = data[data.election < 1997]
    validation = data[data.election == 1997]
    original = training.copy(deep=True)
    model = spec.create_model()
    record = model.train(training, spec.configurations()[0],
                         spec.training_policy.inner_context(validation, PARTIES, {}))
    predictions = validation.drop(columns='winner').copy()
    predictions.loc[predictions.index[0], list(spec.feature_columns)] = np.nan
    if 'country/region' in predictions:
        predictions.loc[predictions.index[1], 'country/region'] = 'unseen'
    probabilities = model.predict_proba(predictions)
    assert probabilities.shape == (5,5)
    assert tuple(model.classes_) == PARTIES
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    assert len(model.predict(predictions)) == 5
    assert model.predict(predictions.iloc[:0]).shape == (0,)
    pd.testing.assert_frame_equal(training, original)
    assert record.training_years == (1987, 1992)


def test_synthetic_end_to_end_all_adapters_without_running_pipeline():
    torch.set_num_threads(1)
    specs = [small_spec(s) for s in default_candidates()]
    result = HistoricalEvaluator(specs, verbose=False).evaluate(sample())
    assert result.report.status == 'complete'
    assert len(result.report.outer_results) == 40
    assert len(result.report.predictions) == 200
    assert len(result.report.scorecard()) == 8
    assert result.report.final_fit['fit_record']['training_years'][-1] == 2019
    for row in result.report.outer_results:
        assert max(row['fit_record']['training_years']) < row['election']
        if row['candidate'].startswith('NN'):
            assert row['fit_record']['validation_year'] is None
            assert row['fit_record']['runs'][0]['duration'] <= 2
            assert row['fit_record']['runs'][0]['best_epoch'] is None


def test_early_stopping_restores_independent_best_checkpoint_and_no_validation_updates():
    torch.set_num_threads(1)
    data = sample()
    training, validation = data[data.election < 1997], data[data.election == 1997]
    encoder = neural.LabelEncoder().fit(PARTIES)
    real_loss = torch.nn.CrossEntropyLoss()
    validation_losses = iter([.8, .2, .3, .4])
    checkpoints = []
    original_step = torch.optim.SGD.step

    class ControlledLoss:
        def __call__(self, logits, targets):
            if torch.is_grad_enabled():
                return real_loss(logits, targets)
            return torch.tensor(next(validation_losses))

    def step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        checkpoints.append([p.detach().clone() for group in optimizer.param_groups for p in group['params']])
        return result

    with patch.object(neural.nn, 'CrossEntropyLoss', return_value=ControlledLoss()), \
         patch.object(torch.optim.SGD, 'step', step):
        net, preprocessor, record = neural._train_network(
            training, validation, encoder, 1, 1000, .1, hidden_sizes=(4,), patience=2)
    assert record['best_epoch'] == 2
    assert record['stopping_epoch'] == 4
    assert len(checkpoints) == 4  # One training batch per epoch; no validation updates.
    for actual, best in zip(net.parameters(), checkpoints[1]):
        torch.testing.assert_close(actual, best)
    assert preprocessor.named_steps['missing_data'].winner_modes_.keys() == {1987,1992}


def test_neural_refit_uses_all_history_and_exact_per_seed_durations():
    torch.set_num_threads(1)
    data = sample().query('election <= 2001')
    model = neural.NeuralNetworkModel()
    config = dict(seeds=(1,2), hidden_sizes=(4,), learning_rate=.01)
    record = model.train(data, config, FitContext(durations={1:2, 2:3}))
    assert [r['stopping_epoch'] for r in record.runs] == [2,3]
    assert all(r['best_epoch'] is None for r in record.runs)
    assert all(set(p.named_steps['missing_data'].winner_modes_) == {1987,1992,1997,2001}
               for p in model.preprocessors)
    before = deepcopy(record)
    model.predict_proba(data.drop(columns='winner'))
    assert model.fit_record == before
    with pytest.raises(ValueError, match='requires a derived duration'):
        neural.NeuralNetworkModel().train(data, config)


def test_validation_boundary_is_enforced_before_neural_fit():
    data = sample().query('election <= 2001')
    with pytest.raises(ValueError, match='precede validation'):
        neural.NeuralNetworkModel().train(data, {}, FitContext(validation=data.query('election == 1997')))


def test_conditional_reuses_stages_only_inside_same_fold():
    from Models.conditional_xgboost_model import Stage
    spec = small_spec(default_candidates()[-1])
    data = sample().query('election < 1997')
    config = spec.configurations()[0]
    context = FitContext()
    fits = []
    original = Stage.fit
    def tracked(stage, features, labels, params):
        fits.append(params.copy())
        return original(stage, features, labels, params)
    with patch.object(Stage, 'fit', tracked):
        first = spec.create_model(); first.train(data, config, context)
        second = spec.create_model(); second.train(data, {**config, 'challenger.max_depth': 3}, context)
        third = spec.create_model(); third.train(data, config, FitContext())
    assert len(fits) == 5  # 2 initial, 1 changed stage, 2 in a fresh fold.
    assert first.change_model is second.change_model
    assert first.destination_model is not second.destination_model
    assert third.change_model is not first.change_model


def test_stage_cache_rejects_a_different_training_dataset():
    spec = small_spec(default_candidates()[-1])
    data = sample().query('election < 1997')
    context = FitContext()
    spec.create_model().train(data, spec.configurations()[0], context)
    with pytest.raises(ValueError, match='different training data'):
        spec.create_model().train(data.assign(con_polling=.8), spec.configurations()[0], context)


@pytest.mark.parametrize('index', [0,1,2,6])
def test_ordinary_models_align_absent_classes_and_single_class_history(index):
    spec = small_spec(default_candidates()[index])
    data = sample().query('election < 1997').assign(winner='lab')
    model = spec.create_model()
    model.train(data, spec.configurations()[0])
    result = model.predict_proba(data.drop(columns='winner'))
    expected = np.zeros((len(data),5)); expected[:,1] = 1
    np.testing.assert_allclose(result, expected)
