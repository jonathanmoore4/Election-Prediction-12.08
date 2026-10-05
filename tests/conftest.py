"""PostgreSQL-backed fixtures; production outputs are never used by tests."""
from uuid import uuid4
import os
import pytest


@pytest.fixture
def tracking_settings(monkeypatch):
    if not os.environ.get('MLFLOW_TRACKING_URI'):
        pytest.skip('Set MLFLOW_TRACKING_URI for the real tracking integration test.')
    return {'experiment_name':'election-tests-' + uuid4().hex}


@pytest.fixture
def postgres_cursor():
    """Rollback-only schema: tests cannot overwrite persisted pipeline runs."""
    from election.sql.apply_sql_queries import connect_database
    from psycopg2 import sql
    if not os.environ.get('ELECTION_DATABASE_URL'):
        pytest.skip('Set ELECTION_DATABASE_URL; see README.md')
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
