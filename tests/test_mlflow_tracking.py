"""Failure, isolation and final neural policy checks against PostgreSQL."""
import json
from pathlib import Path

import pytest

from election.models.evaluation import HistoricalEvaluator
from election.models.tracking import ExecutionTracking, CORE_METRICS
from tests.test_automated_model_selection import history, constant_spec
from tests.test_model_training import sample, small_spec
from election.models.candidates import default_candidates

pytestmark = pytest.mark.usefixtures('tracked_evaluation')


def artifact(tracker, run_id, name):
    return json.loads(Path(tracker.client.download_artifacts(run_id, name)).read_text())


def test_two_executions_keep_hierarchy_and_compact_artifacts():
    specs = [constant_spec(str(i), ['con']) for i in range(8)]
    first = HistoricalEvaluator(specs, verbose=False).evaluate(history()).report
    second = HistoricalEvaluator(specs, verbose=False).evaluate(history()).report
    assert first.execution_id != second.execution_id
    assert len(first.tracking.client.search_runs([first.tracking.experiment_id], max_results=1000)) == 100
    t = first.tracking
    assert t.client.get_run(first.execution_id).info.status == 'FINISHED'
    assert len(t.children(first.execution_id, 'candidate')) == 8
    assert len(t.children(first.execution_id, 'final_test')) == 1
    assert not t.client.get_run(first.execution_id).data.metrics
    for name, run_id in t.candidate_ids.items():
        summary = t.client.get_run(run_id)
        assert set(summary.data.metrics) == {f'mean_{m}' for m in CORE_METRICS}
        assert summary.data.metrics['mean_accuracy'] == .8
        assert len(t.children(run_id, 'election')) == 5
        assert [a.path for a in t.client.list_artifacts(run_id)] == ['candidate_specification.json']
        for r in t.children(run_id, 'election'):
            assert set(r.data.metrics) <= set(CORE_METRICS)
            assert r.data.tags['execution_id'] == first.execution_id
            assert [a.path for a in t.client.list_artifacts(r.info.run_id)] == ['selected_configuration.json']
            selected = artifact(t, r.info.run_id, 'selected_configuration.json')
            assert 'fit_record' not in selected
            assert selected['configuration'] == {'party': 'con'}
    assert first.scorecard().mean_accuracy.tolist() == second.scorecard().mean_accuracy.tolist()


@pytest.mark.parametrize('failure', ['final_evaluation', 'final_logging', 'output'])
def test_final_failure_keeps_outer_results_and_fails_parent(failure, monkeypatch):
    evaluator = HistoricalEvaluator([constant_spec('one', ['con'])], verbose=False)
    t = ExecutionTracking()
    if failure == 'final_evaluation':
        monkeypatch.setattr(evaluator, '_final_evaluation', lambda *a: (_ for _ in ()).throw(RuntimeError('final failure')))
    elif failure == 'final_logging':
        original = t.scores
        def fail(run_id, values, **kw):
            if 'test_rows' in values:
                raise RuntimeError('final failure')
            return original(run_id, values, **kw)
        monkeypatch.setattr(t, 'scores', fail)
    else:
        monkeypatch.setattr('matplotlib.figure.Figure.savefig', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('final failure')))
    with pytest.raises(RuntimeError, match='final failure'):
        evaluator.evaluate(history(), tracking=t)
    assert evaluator.report.status == 'failed'
    assert t.client.get_run(t.parent_id).info.status == 'FAILED'
    assert t.client.get_run(t.parent_id).data.tags['execution_status'] == 'failed'
    assert len(t.election_scores()) == 5
    assert t.children(t.parent_id, 'candidate')[0].info.status == 'FINISHED'
    assert t.children(t.parent_id, 'final_test')[0].info.status == 'FAILED'


@pytest.mark.parametrize('corruption', ['missing', 'nonfinite', 'incomplete', 'out_of_range'])
def test_invalid_current_summary_cannot_select(corruption, monkeypatch):
    t = ExecutionTracking()
    original = t.finish
    def finish(run_id):
        original(run_id)
        run = t.client.get_run(run_id)
        if run.data.tags['kind'] == 'candidate':
            if corruption == 'missing':
                # A client response with a required metric absent.
                pass
            elif corruption == 'incomplete':
                t.client.set_terminated(run_id, 'FAILED')
            else:
                t.client.log_metric(run_id, 'mean_accuracy', float('nan') if corruption == 'nonfinite' else 2)
    monkeypatch.setattr(t, 'finish', finish)
    original_children = t.children
    calls = 0
    def children(parent, kind):
        nonlocal calls
        records = original_children(parent, kind)
        if kind == 'candidate':
            calls += 1
            if corruption == 'missing' and calls == 2:
                records[0].data.metrics.pop('mean_accuracy')
        return records
    monkeypatch.setattr(t, 'children', children)
    evaluator = HistoricalEvaluator([constant_spec('one', ['con'])], verbose=False)
    with pytest.raises(RuntimeError, match='Invalid selection|Incomplete MLflow'):
        evaluator.evaluate(history(), tracking=t)
    assert evaluator.report.selected_candidate is None
    assert t.client.get_run(t.parent_id).info.status == 'FAILED'
    assert not t.children(t.parent_id, 'final_test')


def test_missing_outer_record_cannot_produce_winner(monkeypatch):
    t = ExecutionTracking()
    original = t.children
    def missing(parent, kind):
        records = original(parent, kind)
        return records[:-1] if kind == 'election' else records
    monkeypatch.setattr(t, 'children', missing)
    evaluator = HistoricalEvaluator([constant_spec('one', ['con'])], verbose=False)
    with pytest.raises(RuntimeError, match='Incomplete outer'):
        evaluator.evaluate(history(), tracking=t)
    assert evaluator.report.selected_candidate is None
    assert t.client.get_run(t.parent_id).info.status == 'FAILED'


def test_final_outcomes_cannot_change_selection_or_settings():
    specs = [constant_spec('con', ['con', 'lab']), constant_spec('lab', ['lab'])]
    test = history().query('election == 2019').assign(election=2024)
    a = HistoricalEvaluator(specs, verbose=False).evaluate(history(), test_data=test)
    b = HistoricalEvaluator(specs, verbose=False).evaluate(history(), test_data=test.assign(winner='con'))
    assert a.model_name == b.model_name == 'con'
    assert a.report.final_fit == b.report.final_fit
    assert a.report.final_metrics['accuracy'] == 0
    assert b.report.final_metrics['accuracy'] == 1


def test_real_neural_final_fit_has_same_implementation_and_refit_policy(monkeypatch):
    import torch
    torch.set_num_threads(1)
    spec = small_spec(default_candidates()[3])
    calls = []
    model_class = spec.factory
    original = model_class.train
    def train(model, data, configuration=None, fit_context=None):
        record = original(model, data, configuration, fit_context)
        calls.append((type(model), configuration.copy(), fit_context, record))
        return record
    monkeypatch.setattr(model_class, 'train', train)
    result = HistoricalEvaluator([spec], verbose=False).evaluate(sample())
    refits = [c for c in calls if c[2].validation is None]
    assert len(refits) == 6
    assert all(c[0] is type(result.model) for c in calls)
    assert all(c[1] == refits[0][1] for c in refits)
    assert refits[-1][3].training_years[-1] == 2019
    assert all(c[2].durations == {1: c[3].runs[0]['duration']} for c in refits)
    t = result.report.tracking
    final = t.children(t.parent_id, 'final_test')[0]
    assert artifact(t, final.info.run_id, 'selected_configuration.json')['refit_durations'] == {str(k):v for k,v in refits[-1][2].durations.items()}
    assert not hasattr(result.report, 'inner_results')


def test_logging_failure_at_outer_boundary_prevents_selection(monkeypatch):
    t = ExecutionTracking()
    def fail(*a, **kw):
        raise RuntimeError('logging unavailable')
    monkeypatch.setattr(t, 'scores', fail)
    evaluator = HistoricalEvaluator([constant_spec('one', ['con'])], verbose=False)
    with pytest.raises(RuntimeError, match='logging unavailable'):
        evaluator.evaluate(history(), tracking=t)
    assert evaluator.report.selected_candidate is None
    assert t.client.get_run(t.parent_id).info.status == 'FAILED'
    assert t.children(t.parent_id, 'election') == []
    assert t.children(t.candidate_ids['one'], 'election')[0].info.status == 'FAILED'


def test_undefined_subgroup_is_explicit_and_omitted_from_mean():
    data = history().assign(winner='con', previous_winner=None)
    result = HistoricalEvaluator([constant_spec('one', ['con'])], verbose=False).evaluate(data)
    t = result.report.tracking
    summary = t.client.get_run(t.candidate_ids['one'])
    assert 'mean_changed_seat_accuracy' not in summary.data.metrics
    assert summary.data.tags['undefined.mean_changed_seat_accuracy'] == 'true'
    assert summary.data.params['changed_seat_accuracy_elections'] == '0'
    for run in t.children(summary.info.run_id, 'election'):
        assert run.data.params['changed_seat_evaluation_rows'] == '0'
        assert run.data.tags['undefined.changed_seat_accuracy'] == 'true'
        assert 'changed_seat_accuracy' not in run.data.metrics
