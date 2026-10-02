"""PostgreSQL-backed fixtures; production outputs are never used by tests."""
from uuid import uuid4
import os
import pytest


@pytest.fixture
def tracked_evaluation(tmp_path, monkeypatch):
    if not os.environ.get('MLFLOW_TRACKING_URI'):
        pytest.skip('PostgreSQL-backed tests require MLFLOW_TRACKING_URI; see docs/mlflow.md')
    import election.models.evaluation as evaluation
    monkeypatch.setenv('MLFLOW_EXPERIMENT_NAME', 'election-tests-' + uuid4().hex)
    original = evaluation.HistoricalEvaluator.evaluate

    def evaluate(self, data, **kwargs):
        # Supply a separate tiny final election; no forecasts enter the history.
        kwargs.setdefault('test_data', data.loc[data.election == data.election.max()].assign(election=2024))
        kwargs.setdefault('output_dir', tmp_path)
        return original(self, data, **kwargs)
    monkeypatch.setattr(evaluation.HistoricalEvaluator, 'evaluate', evaluate)
