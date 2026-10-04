"""Small real fits and controlled checkpoint tests for the new adapter contract."""
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

import election.models.custom_model as custom_model_module
import election.models.evaluation as evaluation_module
from election.pipeline.run_pipeline import MODEL_MODULES, model_functions
from importlib import import_module
from election.models.adapters import nn01_model as neural


def model_spec(index):
    model_id = list(MODEL_MODULES)[index]
    function, candidates, metadata = model_functions(model_id)
    module = import_module('election.models.adapters.' + MODEL_MODULES[model_id][0])
    factory_name = ('NeuralNetworkModel' if model_id.startswith('nn') else {
        'xgboost': 'XGBoostModel', 'random_forest': 'RandomForestModel',
        'logistic_regression': 'LogisticRegressionModel',
        'xgboost_expanded': 'XGBoostExpandedModel',
        'conditional_xgboost': 'ConditionalXGBoostModel'}[model_id])
    return dict(model_id=model_id, function=function, candidates=candidates,
                metadata=metadata, factory=getattr(module, factory_name))


def sample():
    columns = set(c for i in range(8) for c in model_spec(i)['metadata']['features'])
    rows = []
    for year in [1987,1992,1997,2001,2005,2010,2015,2017,2019]:
        for i, party in enumerate(custom_model_module.PARTIES):
            rows.append({**{c:.1+.02*i for c in columns}, 'election':year,
                'winner':party, 'previous_winner':custom_model_module.PARTIES[(i+1)%5],
                'country/region':'England', 'incumbent':'con', 'constituency_id':f'{year}-{i}'})
    return pd.DataFrame(rows)


def small_spec(spec):
    config = {**spec['metadata']['fixed_settings'],
              **evaluation_module.parameter_combinations(spec['candidates'])[0]}
    if spec['model_id'].startswith('nn'):
        config.update(max_epochs=2, patience=1, seeds=(1,), learning_rate=.01)
    elif spec['model_id'] == 'conditional_xgboost':
        config.update({'change.n_estimators':2,'challenger.n_estimators':2})
    elif 'n_estimators' in config:
        config['n_estimators'] = 2
    return {**spec, 'configuration':config}


def test_search_sizes_match_original_procedures():
    assert [len(evaluation_module.parameter_combinations(model_spec(i)['candidates']))
            for i in range(8)] == [16,18,4,4,8,1,16,256]


@pytest.mark.parametrize('index', range(8))
def test_every_adapter_exposes_aligned_probabilities_and_keeps_rows(index):
    torch.set_num_threads(1)
    spec = small_spec(model_spec(index)); data = sample()
    training, validation = data[data.election < 1997], data[data.election == 1997]
    original = training.copy(deep=True)
    context = custom_model_module.fit_context(
        validation=validation if spec['model_id'].startswith('nn') else None)
    model = spec['factory']()
    record = model.train(training, spec['configuration'], context)
    predictions = validation.drop(columns='winner').copy()
    predictions.loc[predictions.index[0], spec['metadata']['features']] = np.nan
    predictions.loc[predictions.index[1], 'country/region'] = 'unseen'
    probabilities = model.predict_proba(predictions)
    assert probabilities.shape == (5,5)
    assert tuple(model.classes_) == custom_model_module.PARTIES
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1),1,atol=1e-6)
    assert model.predict(predictions.iloc[:0]).shape == (0,)
    pd.testing.assert_frame_equal(training,original)
    assert record['training_years'] == (1987,1992)


@pytest.mark.parametrize('index', range(8))
def test_all_model_functions_run_nested_cv_without_logging_or_fitted_result_objects(index):
    torch.set_num_threads(1)
    spec = small_spec(model_spec(index)); data = sample()
    result = evaluation_module.nested_cv(spec['function'], {}, data,
        model_id=spec['model_id'], metadata={**spec['metadata'], 'fixed_settings':spec['configuration']},
        verbose=False)
    assert result['status'] == 'complete'
    assert len(result['outer_results']) == 5
    assert result['run_id'] is None
    assert all(max(fold['training_years']) < year for year,fold in result['outer_results'].items())
    if spec['model_id'].startswith('nn'):
        assert all(fold['refit_durations'] in ({1:1},{1:2}) for fold in result['outer_results'].values())


@pytest.mark.parametrize('index', range(8))
def test_model_functions_reject_unknown_hyperparameters(index):
    spec = model_spec(index); data = sample()
    with pytest.raises(ValueError,match='unsupported hyperparameters'):
        spec['function'](data[data.election<1997],data[data.election==1997],{'misspelt_parameter':1})


def test_early_stopping_restores_independent_best_checkpoint_and_no_validation_updates():
    torch.set_num_threads(1)
    data = sample()
    training, validation = data[data.election < 1997], data[data.election == 1997]
    encoder = neural.LabelEncoder().fit(custom_model_module.PARTIES)
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
    record = model.train(data, config, custom_model_module.fit_context(durations={1:2, 2:3}))
    assert [r['stopping_epoch'] for r in record['runs']] == [2,3]
    assert all(r['best_epoch'] is None for r in record['runs'])
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
        neural.NeuralNetworkModel().train(data, {}, custom_model_module.fit_context(validation=data.query('election == 1997')))


def test_conditional_reuses_stages_only_inside_same_fold():
    import election.models.adapters.conditional_xgboost_model as conditional_xgboost_model_module
    spec = small_spec(model_spec(7))
    data = sample().query('election < 1997')
    config = spec['configuration']
    context = custom_model_module.fit_context()
    fits = []
    original = conditional_xgboost_model_module.Stage.fit
    def tracked(stage, features, labels, params):
        fits.append(params.copy())
        return original(stage, features, labels, params)
    with patch.object(conditional_xgboost_model_module.Stage, 'fit', tracked):
        first = spec['factory'](); first.train(data, config, context)
        second = spec['factory'](); second.train(data, {**config, 'challenger.max_depth': 3}, context)
        third = spec['factory'](); third.train(data, config, custom_model_module.fit_context())
    assert len(fits) == 5  # 2 initial, 1 changed stage, 2 in a fresh fold.
    assert first.change_model is second.change_model
    assert first.destination_model is not second.destination_model
    assert third.change_model is not first.change_model


def test_stage_cache_rejects_a_different_training_dataset():
    spec = small_spec(model_spec(7))
    data = sample().query('election < 1997')
    context = custom_model_module.fit_context()
    spec['factory']().train(data, spec['configuration'], context)
    with pytest.raises(ValueError, match='different training data'):
        spec['factory']().train(data.assign(con_polling=.8), spec['configuration'], context)


@pytest.mark.parametrize('index', [0,1,2,6])
def test_ordinary_models_align_absent_classes_and_single_class_history(index):
    spec = small_spec(model_spec(index))
    data = sample().query('election < 1997').assign(winner='lab')
    model = spec['factory']()
    model.train(data, spec['configuration'])
    result = model.predict_proba(data.drop(columns='winner'))
    expected = np.zeros((len(data),5)); expected[:,1] = 1
    np.testing.assert_allclose(result, expected)


def test_neural_outer_2017_keeps_original_inner_validation_and_median_refit(monkeypatch):
    torch.set_num_threads(1)
    spec = small_spec(model_spec(3)); data = sample()
    calls = []
    original = neural.NeuralNetworkModel.train
    def train(model, rows, configuration=None, fit_context=None):
        record = original(model, rows, configuration, fit_context)
        calls.append(dict(training_years=record['training_years'],
                          validation_year=record['validation_year'],
                          durations=deepcopy(fit_context['durations']), runs=record['runs']))
        return record
    monkeypatch.setattr(neural.NeuralNetworkModel, 'train', train)
    schedule = dict(name='2017-regression', purpose='model_comparison',
        outer_elections=[2017], inner_elections={2017:[1997,2001,2005,2010,2015]},
        training_cutoff=None)
    result = evaluation_module.nested_cv(spec['function'], {}, data, model_id='nn01',
        metadata={**spec['metadata'],'fixed_settings':spec['configuration']},
        schedule=schedule, verbose=False)
    inner, outer = calls[:-1], calls[-1]
    assert [call['validation_year'] for call in inner] == [1997,2001,2005,2010,2015]
    assert all(max(call['training_years']) < call['validation_year'] for call in inner)
    assert outer['validation_year'] is None
    assert outer['training_years'][-1] == 2015
    expected = neural.median_refit_durations([{'runs':call['runs']} for call in inner])
    assert outer['durations'] == result['outer_results'][2017]['refit_durations'] == expected
    assert outer['runs'][0]['duration'] == expected[1]


def test_neural_final_outcomes_never_enter_training_or_change_refit_duration(monkeypatch):
    from election.models.config import FINAL_EVALUATION_SCHEDULE
    torch.set_num_threads(1)
    spec = small_spec(model_spec(3)); data = sample()
    test = data.query('election == 2019').assign(election=2024)
    final_a = pd.concat([data,test],ignore_index=True)
    final_b = final_a.copy(); final_b.loc[final_b.election==2024,'winner']='con'
    calls=[]; original=neural.NeuralNetworkModel.train
    def train(model,rows,configuration=None,fit_context=None):
        assert rows.election.max() <= 2019
        if fit_context['validation'] is not None:
            assert fit_context['validation'].election.max() <= 2019
        calls.append(fit_context['validation'] is None)
        return original(model,rows,configuration,fit_context)
    monkeypatch.setattr(neural.NeuralNetworkModel,'train',train)
    results=[evaluation_module.nested_cv(spec['function'],{},frame,model_id='nn01',
        metadata={**spec['metadata'],'fixed_settings':spec['configuration']},
        schedule=FINAL_EVALUATION_SCHEDULE,verbose=False) for frame in (final_a,final_b)]
    assert calls.count(True)==2
    assert results[0]['outer_results'][2024]['refit_durations']==results[1]['outer_results'][2024]['refit_durations']
    assert results[0]['outer_results'][2024]['hyperparameters']==results[1]['outer_results'][2024]['hyperparameters']
