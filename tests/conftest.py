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


@pytest.fixture
def postgres_cursor():
    """Rollback-only schema: tests cannot overwrite persisted pipeline runs."""
    from election.sql.apply_sql_queries import connect_database
    from psycopg2 import sql
    if not os.environ.get('ELECTION_DATABASE_URL'):
        pytest.skip('Set ELECTION_DATABASE_URL; see docs/postgresql.md')
    connection = connect_database()
    try:
        with connection.cursor() as cursor:
            schema = 'test_' + uuid4().hex
            cursor.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
            cursor.execute(sql.SQL('SET LOCAL search_path TO {}, pg_catalog').format(sql.Identifier(schema)))
            yield cursor
    finally:
        connection.rollback()
        connection.close()
