"""Exercise nested selection and leakage boundaries without expensive estimators."""
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from Models.candidates import CandidateSpec
from Models.custom_model import custom_model, FitContext, FitRecord, PARTIES
from Models.evaluation import HistoricalEvaluator, metrics, OUTER_ELECTIONS
from Models.training_policy import NeuralTrainingPolicy

selection = import_module('Run Pipeline.additional_funcs.automated_model_selection')


def history():
    rows = []
    for year in (1987, 1992, 1997, 2001, *OUTER_ELECTIONS):
        # 2019 has more seats, deliberately testing equal election weights.
        for seat in range(20 if year == 2019 else 4):
            rows.append(dict(election=year, winner='con' if year < 2019 else 'lab',
                             previous_winner='con', constituency_id=f'{year}-{seat}'))
    data = pd.DataFrame(rows)
    data.index = np.arange(len(data)) * 3
    return data


class ConstantModel(custom_model):
    instances = []

    def __init__(self, name='constant'):
        super().__init__(name)
        self.instances.append(self)
        self.calls = 0

    def train(self, data, configuration=None, fit_context=None):
        self.calls += 1
        assert self.calls == 1, 'Each fold must use a fresh object'
        context = fit_context or FitContext()
        assert context.validation is None
        self.training_data = data.copy()
        return self._record(data, configuration, context)

    def predict_proba(self, data):
        assert 'winner' not in data
        assert data.election.min() > self.training_data.election.max()
        probabilities = np.zeros((len(data), len(self.classes_)))
        probabilities[:, list(self.classes_).index(self.hyperparameters['party'])] = 1
        return probabilities


def constant_spec(name, parties):
    return CandidateSpec(name, lambda: ConstantModel(name), {'party': parties})


def test_five_elections_equal_weights_fresh_objects_and_probability_audit():
    ConstantModel.instances.clear()
    data = history()
    original = data.copy(deep=True)
    specs = [constant_spec('usually correct', ['con']), constant_spec('2019 specialist', ['lab'])]
    with TemporaryDirectory() as directory:
        result = selection.automated_model_selection(data, Path(directory)/'scores.csv',
                                                    candidates=specs, verbose=False)
        assert result.model.name == 'usually correct'
        card = result.report.scorecard()
        assert card.mean_accuracy.tolist() == [0.8, 0.2]
        assert len(result.report.outer_results) == 10
        assert len(result.report.predictions) == 2 * len(data[data.election.isin(OUTER_ELECTIONS)])
        saved = pd.read_csv(Path(directory)/'scores_predictions.csv')
        np.testing.assert_allclose(saved.filter(like='probability_').sum(axis=1), 1)
        import json
        audit = json.loads((Path(directory)/'scores_report.json').read_text())
        assert audit['status'] == 'complete'
        assert audit['metadata']['data_version']
        assert audit['metadata']['code_version']
        assert result.report.final_fit['fit_record']['training_years'][-1] == 2019
        assert any(r['reused'] for r in result.report.inner_results)
    pd.testing.assert_frame_equal(data, original)
    assert all(m.calls == 1 for m in ConstantModel.instances)
    assert result.model is ConstantModel.instances[-1]
    legacy_model, legacy_name = result
    assert legacy_model is result.model and legacy_name == 'usually correct'


def test_config_selected_by_inner_accuracy_and_outer_changes_cannot_change_earlier_fit():
    spec = constant_spec('search', ['lab', 'con'])
    data = history()
    before = HistoricalEvaluator([spec], verbose=False).evaluate(data)
    data.loc[data.election == 2019, 'winner'] = 'con'
    after = HistoricalEvaluator([spec], verbose=False).evaluate(data)
    assert all(r['configuration']['party'] == 'con' for r in before.report.outer_results)
    assert [r['configuration'] for r in before.report.outer_results] == [r['configuration'] for r in after.report.outer_results]
    assert all(
        a['predicted_winner'] == b['predicted_winner']
        for a, b in zip(before.report.predictions, after.report.predictions))


def test_ties_use_registry_and_configuration_order():
    # All validation winners are con; identical con searches tie exactly.
    data = history().assign(winner='con')
    specs = [constant_spec('first', ['con']), constant_spec('second', ['con'])]
    result = HistoricalEvaluator(specs, verbose=False).evaluate(data)
    assert result.model.name == 'first'


def test_missing_elections_and_future_outcomes_fail_before_training():
    ConstantModel.instances.clear()
    evaluator = HistoricalEvaluator([constant_spec('test', ['con'])], verbose=False)
    with pytest.raises(ValueError, match='outer evaluation'):
        evaluator.evaluate(history().query('election != 2010'))
    assert not ConstantModel.instances
    with pytest.raises(ValueError, match='forecast election'):
        evaluator.evaluate(pd.concat([history(), history().iloc[:1].assign(election=2024)]))
    with pytest.raises(ValueError, match='missing'):
        evaluator.evaluate(history().query('election != 1997'))


def test_failed_fit_does_not_select_or_average_incomplete_results():
    class BrokenModel(ConstantModel):
        def train(self, data, configuration=None, fit_context=None):
            if data.election.max() >= 2005:
                raise RuntimeError('deliberate failure')
            return super().train(data, configuration, fit_context)
    spec = CandidateSpec('broken', lambda: BrokenModel('broken'), {'party': ['con']})
    with TemporaryDirectory() as directory:
        evaluator = HistoricalEvaluator([spec], scores_path=Path(directory)/'scores.csv', verbose=False)
        with pytest.raises(RuntimeError, match='deliberate failure'):
            evaluator.evaluate(history())
        assert evaluator.report.status == 'failed'
        assert evaluator.report.selected_candidate is None
        assert evaluator.report.scorecard().empty
        assert (Path(directory)/'scores_report.json').exists()


def test_metrics_keep_unknown_previous_winner_and_empty_slices_explicit():
    frame = pd.DataFrame(dict(winner=['con', 'lab'], previous_winner=['con', None]))
    probs = np.zeros((2, 5)); probs[:, 0] = 1
    result = metrics(frame, probs, PARTIES)
    assert result['changed_seat_accuracy'] is None
    assert result['changed_seat_evaluation_rows'] == 0
    assert result['retained_seat_accuracy'] == 1
    assert result['evaluation_rows'] == 2
    assert result['previous_winner_accuracy'] == .5


def test_median_uses_selected_checkpoint_epochs_per_seed_rounding_halves_up():
    records = [FitRecord((1987, 1992), {}, v, [
        dict(seed=1, best_epoch=a, stopping_epoch=900),
        dict(seed=2, best_epoch=b, stopping_epoch=900)])
        for v, a, b in [(1997, 2, 10), (2001, 3, 40)]]
    context = NeuralTrainingPolicy().refit_context(records, PARTIES)
    assert context.validation is None
    assert context.durations == {1: 3, 2: 25}
    assert records[0].runs[0]['best_epoch'] == 2


def test_retrain_compatibility_returns_new_object_without_mutating_old():
    spec = constant_spec('test', ['con', 'lab'])
    evaluator = HistoricalEvaluator([spec], verbose=False)
    old = evaluator.tune_and_refit(spec, history().query('election < 2019'))
    before = old.training_data.copy(deep=True)
    new = old.retrain(history(), spec=spec)
    assert new is not old
    pd.testing.assert_frame_equal(old.training_data, before)
    assert new.training_data.election.max() == 2019


def test_missing_latest_inner_election_cannot_be_silently_skipped():
    spec = constant_spec('test', ['con'])
    evaluator = HistoricalEvaluator([spec], verbose=False)
    with pytest.raises(ValueError, match='2017'):
        evaluator.tune_and_refit(spec, history().query('election < 2017'), forecast_election=2019)


def test_nullable_previous_winner_and_unknown_outcomes_do_not_drop_prediction_rows():
    frame = pd.DataFrame(dict(winner=['con', 'lab'],
                              previous_winner=pd.Series(['con', pd.NA], dtype='string')))
    probabilities = np.zeros((2,5)); probabilities[:,0] = 1
    assert metrics(frame, probabilities, PARTIES)['previous_winner_accuracy'] == .5


def test_invalid_probabilities_abort_comparison():
    class BadProbabilities(ConstantModel):
        def predict_proba(self, data):
            return np.full((len(data),5), .3)
    spec = CandidateSpec('bad', lambda: BadProbabilities('bad'), {'party':['con']})
    evaluator = HistoricalEvaluator([spec], verbose=False)
    with pytest.raises(ValueError, match='sum to one'):
        evaluator.evaluate(history())
    assert evaluator.report.status == 'failed'
    assert evaluator.report.metadata['active_fit']['validation_election'] == 1997


def test_inner_accuracy_selects_only_that_configurations_duration_records():
    contexts = []
    class CheckpointModel(ConstantModel):
        def train(self, data, configuration=None, fit_context=None):
            self.calls += 1
            assert self.calls == 1
            self.training_data = data.copy()
            if fit_context.validation is not None:
                # Deliberately favour the WRONG configuration by loss.
                right = configuration['party'] == 'con'
                runs = [dict(seed=1, best_epoch=2 if right else 99,
                             best_validation_loss=10 if right else .001)]
            else:
                contexts.append(deepcopy(fit_context.durations))
                runs = []
            return self._record(data, configuration, fit_context, runs)
    from copy import deepcopy
    spec = CandidateSpec('checkpoint', lambda: CheckpointModel('checkpoint'),
                         {'party':['lab','con']}, training_policy=NeuralTrainingPolicy())
    result = HistoricalEvaluator([spec], verbose=False).evaluate(history())
    assert result.model.hyperparameters['party'] == 'con'
    assert contexts == [{1:2}] * 6  # Five outer refits and the forecast refit.


def test_no_outcome_or_identifier_columns_are_passed_as_predictors():
    class StrictModel(ConstantModel):
        def train(self, data, configuration=None, fit_context=None):
            assert set(data) == {'election','winner'}
            return super().train(data, configuration, fit_context)
        def predict_proba(self, data):
            assert set(data) == {'election'}
            return super().predict_proba(data)
    spec = CandidateSpec('strict', lambda: StrictModel('strict'), {'party':['con']})
    result = HistoricalEvaluator([spec], verbose=False).evaluate(
        history().assign(winning_party_vote_share=.7, other_outcome=.6))
    assert result.report.status == 'complete'
