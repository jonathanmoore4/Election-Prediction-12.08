"""Election boundaries and deterministic nested tuning without expensive fits."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from election.models.config import MODEL_COMPARISON_SCHEDULE, FINAL_EVALUATION_SCHEDULE
from election.models import evaluation
from election.models.adapters.nn01_model import median_refit_durations


def history():
    return pd.DataFrame([dict(election=year, winner='con' if year < 2019 else 'lab',
                             previous_winner='con', constituency_id=f'{year}-{seat}')
        for year in (1987, 1992, 1997, 2001, 2005, 2010, 2015, 2017, 2019)
        for seat in range(20 if year == 2019 else 4)])


def constant(train, test, parameters):
    assert train.election.max() < test.election.min()
    return float(test.winner.eq(parameters['party']).mean())


def evaluated(model_id='constant', candidates=None, **kwargs):
    return evaluation.nested_cv(constant, candidates or {'party': ['con', 'lab']},
        history(), model_id=model_id, verbose=False, **kwargs)


def test_nested_selection_is_equal_election_weighted_and_does_not_require_2024():
    result = evaluated()
    assert isinstance(result, dict)
    assert result['mean_outer_accuracy'] == .8
    assert result['run_id'] is None
    assert len(result['outer_results']) == 5
    assert all(fold['hyperparameters'] == {'party': 'con'} for fold in result['outer_results'].values())
    assert result['outer_results'][2019]['accuracy'] == 0


def test_every_fit_uses_earlier_data_and_selected_parameters_with_cached_inner_fits():
    calls = []
    def model(train, test, parameters, *, fit_records=None, return_details=False, cache=None):
        calls.append((test.election.iloc[0], tuple(train.election.unique()), parameters.copy(), fit_records))
        return constant(train, test, parameters)
    result = evaluation.nested_cv(model, {'party': ['con', 'lab']}, history(),
                                  model_id='test', verbose=False)
    outer = [call for call in calls if call[3] is not None]
    assert len(outer) == 5
    assert outer[-1][:3] == (2019, (1987,1992,1997,2001,2005,2010,2015,2017), {'party':'con'})
    inner = [call for call in calls if call[3] is None]
    assert len(inner) == 12  # six distinct elections, two configurations
    assert all(max(years) < year for year, years, _, _ in calls)
    assert result['status'] == 'complete'


def test_configuration_tie_keeps_existing_parametergrid_order():
    tuned = evaluation.tune_hyperparameters(lambda *args: .5, {'party': ['lab', 'con']},
        history(), [1997,2001], verbose=False)
    assert tuned['best_hyperparameters'] == {'party':'lab'}
    assert evaluation.parameter_combinations({}) == [{}]


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -1, 2, True, None])
def test_invalid_scores_fail_instead_of_averaging_fewer_elections(score):
    with pytest.raises(ValueError, match='finite accuracy'):
        evaluation.nested_cv(lambda *args: score, {}, history(), model_id='bad', verbose=False)


def test_model_failure_is_not_silently_omitted():
    def broken(train, test, parameters):
        if test.election.iloc[0] == 2001:
            raise RuntimeError('training failed')
        return .5
    with pytest.raises(RuntimeError, match='training failed'):
        evaluation.tune_hyperparameters(broken, {}, history(), [1997,2001], verbose=False)


@pytest.mark.parametrize('change', ['missing_outer', 'missing_inner', 'future_inner', 'bad_cutoff', 'no_history'])
def test_schedule_validation_rejects_invalid_boundaries(change):
    data, schedule = history(), deepcopy(MODEL_COMPARISON_SCHEDULE)
    if change == 'missing_outer': data = data[data.election != 2019]
    if change == 'missing_inner': data = data[data.election != 1997]
    if change == 'future_inner': schedule['inner_elections'][2005] = [1997, 2010]
    if change == 'bad_cutoff': schedule['training_cutoff'] = 1992
    if change == 'no_history': data = data[data.election >= 1997]
    with pytest.raises(ValueError): evaluation.validate_schedule(data, schedule)


def test_final_tunes_once_on_pre2019_history_and_outcomes_do_not_change_settings(monkeypatch):
    original = evaluation.tune_hyperparameters
    calls = []
    def tune(*args, **kwargs):
        calls.append(args[2].election.max())
        return original(*args, **kwargs)
    monkeypatch.setattr(evaluation, 'tune_hyperparameters', tune)
    data = history()
    a = pd.concat([data, data.query('election == 2019').assign(election=2024)], ignore_index=True)
    b = a.copy(); b.loc[b.election == 2024, 'winner'] = 'con'
    results = [evaluation.nested_cv(constant, {'party':['con','lab']}, frame,
        model_id='constant', schedule=FINAL_EVALUATION_SCHEDULE, verbose=False) for frame in (a,b)]
    assert calls == [2019,2019]
    assert results[0]['outer_results'][2024]['hyperparameters'] == results[1]['outer_results'][2024]['hyperparameters']
    assert results[0]['mean_outer_accuracy'] == 0
    assert results[1]['mean_outer_accuracy'] == 1
    assert results[0]['outer_results'][2024]['training_years'][-1] == 2019


def test_original_neural_median_rule_keeps_seed_specific_durations_and_half_up():
    records = [dict(runs=[dict(seed=1,best_epoch=a),dict(seed=2,best_epoch=b)])
               for a,b in [(2,4),(3,7)]]
    assert median_refit_durations(records) == {1:3,2:6}
    with pytest.raises(ValueError): median_refit_durations([])
    with pytest.raises(ValueError): median_refit_durations([dict(runs=[dict(seed=1,best_epoch=0)])])


def test_current_outer_outcomes_cannot_change_its_inner_scores_or_parameters():
    a = evaluated()
    data = history(); data.loc[data.election == 2017,'winner'] = 'lab'
    b = evaluation.nested_cv(constant, {'party':['con','lab']}, data, model_id='constant', verbose=False)
    assert a['outer_results'][2017]['inner_accuracies'] == b['outer_results'][2017]['inner_accuracies']
    assert a['outer_results'][2017]['hyperparameters'] == b['outer_results'][2017]['hyperparameters']


def test_no_mlflow_record_is_created_for_partial_evaluation(monkeypatch):
    from election.models import tracking
    calls=[]
    monkeypatch.setattr(tracking,'log_evaluation_result',lambda *a,**kw: calls.append(a))
    def broken(train,test,parameters):
        if test.election.iloc[0] == 2019: raise RuntimeError('outer failure')
        return .5
    with pytest.raises(RuntimeError):
        evaluation.nested_cv(broken,{},history(),model_id='bad',tracking={},verbose=False)
    assert calls == []


def test_fixed_settings_are_applied_to_every_fit_and_recorded_once_in_metadata():
    calls=[]
    def model(train,test,parameters):
        calls.append(dict(parameters))
        assert parameters['random_state']==42
        return constant(train,test,parameters)
    result=evaluation.nested_cv(model,{'party':['con','lab']},history(),model_id='constant',
        metadata={'fixed_settings':{'random_state':42}},verbose=False)
    assert calls
    assert result['metadata']['fixed_settings']=={'random_state':42}
    assert all(fold['hyperparameters']=={'party':'con'} for fold in result['outer_results'].values())
