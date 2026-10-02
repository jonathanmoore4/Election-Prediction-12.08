"""Inexpensive synthetic tracking demonstration, including service restart."""
from pathlib import Path
from uuid import uuid4
import json
import os
import subprocess
from importlib import import_module
from unittest.mock import patch

import numpy as np
import pandas as pd

from election.models.candidates import CandidateSpec
from election.models.custom_model import custom_model, PARTIES
from election.models.training_policy import NeuralTrainingPolicy
from election.models.tracking import CORE_METRICS, ExecutionTracking
from election.models.evaluation import OUTER_ELECTIONS
from election import paths


class DemoModel(custom_model):
    """Constant probabilities with controlled fold epochs; no neural training."""
    def __init__(self, name):
        super().__init__(name)

    def train(self, data, configuration=None, fit_context=None):
        assert data.election.max() < 2024
        if fit_context.validation is not None:
            runs = [dict(seed=seed, best_epoch=2 + seed) for seed in configuration['seeds']]
        else:
            assert fit_context.durations == {1: 3, 2: 4}
            runs = [dict(seed=seed, duration=duration) for seed, duration in fit_context.durations.items()]
        return self._record(data, configuration, fit_context, runs)

    def predict_proba(self, data):
        assert 'winner' not in data
        probabilities = np.full((len(data), len(PARTIES)), .025)
        probabilities[:, 0] = .9
        return probabilities


def demonstration():
    root = paths.project_root()
    name = 'mlflow-validation-' + uuid4().hex[:10]
    os.environ['MLFLOW_EXPERIMENT_NAME'] = name
    candidates = [CandidateSpec(f'Demo {i+1}', lambda i=i: DemoModel(f'Demo {i+1}'),
        search_space={'learning_rate': [.01, .02]},
        fixed_settings={'seeds': (1, 2), 'hidden_sizes': (4,)},
        training_policy=NeuralTrainingPolicy()) for i in range(8)]
    train = pd.DataFrame([dict(election=year, winner='con', previous_winner=previous)
        for year in (1987, 1992, 1997, 2001, *OUTER_ELECTIONS)
        for previous in ('con', 'lab')])
    test = train.query('election == 2019').assign(election=2024)
    pipeline = import_module('election.pipeline.run_pipeline')
    outputs = root / '.mlflow-demo' / name
    reports = []
    # Exercise the real pipeline, substituting only remote data acquisition/preparation.
    for execution in (1, 2):
        output = outputs / str(execution)
        with patch.object(pipeline.read_in_raw, 'read_raw_data', return_value={}), \
             patch.object(pipeline.clean_data_all, 'clean_all_data', return_value={}), \
             patch.object(pipeline.apply_sql_queries, 'apply_sql_queries', return_value={'train': train, 'test': test}):
            result = pipeline.run_pipeline(output_dir=output, candidates=candidates)
        report = result.report
        assert report.status == 'complete'
        tracker = report.tracking
        parent = tracker.client.get_run(report.execution_id)
        assert parent.info.status == 'FINISHED'
        assert not parent.data.metrics
        assert len(tracker.children(report.execution_id, 'candidate')) == 8
        assert len(tracker.children(report.execution_id, 'final_test')) == 1
        count = 0
        for run_id in tracker.candidate_ids.values():
            summary = tracker.client.get_run(run_id)
            assert set(summary.data.metrics) == {f'mean_{m}' for m in CORE_METRICS}
            count += len(summary.data.metrics)
            evaluations = tracker.children(run_id, 'election')
            assert len(evaluations) == 5
            for run in evaluations:
                assert set(run.data.metrics) == set(CORE_METRICS)
                assert run.data.params['evaluation_rows'] == '2'
                assert run.data.params['changed_seat_evaluation_rows'] == '1'
                assert run.data.params['retained_seat_evaluation_rows'] == '1'
                assert run.data.tags['execution_id'] == report.execution_id
                selected = read_artifact(tracker, run.info.run_id, 'selected_configuration.json')
                assert selected['refit_durations'] == {'1': 3, '2': 4}
                assert selected['configuration']['learning_rate'] == .01
                assert 'fit_record' not in selected
                assert [a.path for a in tracker.client.list_artifacts(run.info.run_id)] == ['selected_configuration.json']
                count += len(run.data.metrics)
            assert [a.path for a in tracker.client.list_artifacts(run_id)] == ['candidate_specification.json']
        final = tracker.children(report.execution_id, 'final_test')[0]
        count += len(final.data.metrics)
        assert count == 245
        assert final.data.tags['candidate_specification_run_id'] == tracker.candidate_ids['Demo 1']
        for filename in ('train.csv', 'test.csv', 'predictor_descriptions.md',
                         'test_predictions.csv', 'test_confusion_matrix.csv', 'test_confusion_matrix.png'):
            assert (output / filename).exists()
        assert not list(output.glob('model_accuracies*'))
        reports.append(report)
    assert reports[0].execution_id != reports[1].execution_id
    assert len(reports[0].tracking.client.search_runs([reports[0].tracking.experiment_id], max_results=1000)) == 100
    first = reports[0]
    original = read_artifact(first.tracking, first.tracking.candidate_ids['Demo 1'], 'candidate_specification.json')
    subprocess.run([str(root / 'scripts/mlflow-services.sh'), 'restart'], check=True)
    assert first.tracking.client.get_run(first.execution_id).info.status == 'FINISHED'
    assert read_artifact(first.tracking, first.tracking.candidate_ids['Demo 1'], 'candidate_specification.json') == original
    assert len(first.scorecard()) == 8
    print(f'Validated two independent executions, 245 scores each, outputs and restart persistence in {name}.')
    print('Execution IDs:', ', '.join(r.execution_id for r in reports))


def read_artifact(tracker, run_id, filename):
    return json.loads(Path(tracker.client.download_artifacts(run_id, filename)).read_text())


if __name__ == '__main__':
    demonstration()
