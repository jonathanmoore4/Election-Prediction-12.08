"""Build persistent, isolated PostgreSQL snapshots from cleaned DataFrames."""
from datetime import datetime, timezone
from io import StringIO
import logging
import os
import shlex
from pathlib import Path
from uuid import uuid4

import pandas as pd
from pandas.api import types
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json
from election import paths

SQL_DIR = Path(__file__).resolve().parent
SQL_FILES = [SQL_DIR / name for name in (
    '1_reading_in_csv_files.sql', '2_adding_previous_elections.sql',
    '3_add_polling.sql', '4_projected_polling.sql',
    '5_further_feature_engineering.sql', '6_reading_into_testtrain.sql',
)]
INPUT_NAMES = frozenset(('historical', 'notional_1992', 'notional_2001',
                         'notional_2005', 'notional_2019', 'results_2024', 'polling'))
logger = logging.getLogger(__name__)


def connect_database(database_url=None):
    """Resolve the pipeline DSN from arguments, environment, or local setup."""
    url = database_url or os.environ.get('ELECTION_DATABASE_URL')
    if not url:
        config = paths.project_root() / '.env.election'
        if config.is_file():
            # Read the setup script's shell-quoted assignment without executing
            # shell commands or changing the notebook's process environment.
            for line in config.read_text().splitlines():
                tokens = shlex.split(line, comments=True)
                if tokens and tokens[0] == 'export':
                    tokens = tokens[1:]
                if len(tokens) == 1 and tokens[0].startswith('ELECTION_DATABASE_URL='):
                    url = tokens[0].partition('=')[2]
    if not url:
        raise ValueError('Set ELECTION_DATABASE_URL to the PostgreSQL election database; see README.md')
    return psycopg2.connect(url, connect_timeout=10, application_name='election-prediction')


def load_dataframe(cursor, name, dataframe):
    """Create a typed table and bulk-load it, preserving NULLs and column case."""
    if dataframe.columns.empty or not dataframe.columns.is_unique:
        raise ValueError('Input DataFrames must have nonempty, unique columns')
    definitions = []
    for column, dtype in dataframe.dtypes.items():
        if types.is_bool_dtype(dtype):
            pg_type = 'BOOLEAN'
        elif types.is_integer_dtype(dtype):
            pg_type = 'BIGINT'
        elif types.is_numeric_dtype(dtype):
            pg_type = 'DOUBLE PRECISION'
        elif types.is_datetime64_any_dtype(dtype):
            pg_type = 'TIMESTAMP WITH TIME ZONE' if isinstance(dtype, pd.DatetimeTZDtype) else 'TIMESTAMP'
        else:
            pg_type = 'TEXT'
        definitions.append(sql.SQL('{} {}').format(sql.Identifier(str(column)), sql.SQL(pg_type)))
    cursor.execute(sql.SQL('CREATE TABLE {} ({})').format(
        sql.Identifier(name), sql.SQL(', ').join(definitions)))
    # A fresh token avoids treating real empty strings (or literal "NULL") as NULL.
    null_token = '__null_' + uuid4().hex[:8]
    while dataframe.astype(str).eq(null_token).any().any():
        null_token = '__null_' + uuid4().hex[:8]
    buffer = StringIO()
    dataframe.to_csv(buffer, index=False, header=False, na_rep=null_token)
    buffer.seek(0)
    statement = sql.SQL('COPY {} ({}) FROM STDIN WITH (FORMAT CSV, NULL {})').format(
        sql.Identifier(name), sql.SQL(', ').join(sql.Identifier(str(c)) for c in dataframe.columns),
        sql.Literal(null_token))
    cursor.copy_expert(statement.as_string(cursor), buffer)
    cursor.execute(sql.SQL('ANALYZE {}').format(sql.Identifier(name)))


def read_dataframe(cursor, query):
    cursor.execute(query)
    description = cursor.description
    frame = pd.DataFrame.from_records(cursor.fetchall(), columns=[c.name for c in description])
    # All-NULL float columns must remain numeric, rather than object/None.
    for column in description:
        if column.type_code in (700, 701):
            frame[column.name] = pd.to_numeric(frame[column.name]).astype(float)
    return frame


def apply_sql_queries(cleaned_data, *, database_url=None):
    """Commit a new run schema atomically; failures leave no partial snapshot.

    Successful schemas are retained for inspection. Returned DataFrames carry
    the schema name in attrs['database_schema']; credentials are never exported.
    """
    if set(cleaned_data) != INPUT_NAMES:
        raise ValueError(f'Expected cleaned datasets: {", ".join(sorted(INPUT_NAMES))}')
    schema = 'run_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_') + uuid4().hex[:12]
    connection = connect_database(database_url)
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
                cursor.execute(sql.SQL('SET LOCAL search_path TO {}, pg_catalog').format(sql.Identifier(schema)))
                for name, dataframe in cleaned_data.items():
                    logger.info('Loading %s: %s rows', name, len(dataframe))
                    load_dataframe(cursor, 'input_' + name, dataframe)
                for sql_file in SQL_FILES:
                    logger.info('Running %s', sql_file.name)
                    cursor.execute(sql_file.read_text())
                result = {name: read_dataframe(cursor, sql.SQL(
                    'SELECT * FROM {} ORDER BY election, constituency_id, constituency_name'
                ).format(sql.Identifier(name + '_data'))) for name in ('train', 'test')}
                cursor.execute('CREATE TABLE snapshot_info (completed_at TIMESTAMPTZ NOT NULL, '
                               'input_rows JSONB NOT NULL, output_rows JSONB NOT NULL, sql_files JSONB NOT NULL)')
                cursor.execute('INSERT INTO snapshot_info VALUES (clock_timestamp(), %s, %s, %s)', (
                    Json({name: len(frame) for name, frame in cleaned_data.items()}),
                    Json({name: len(frame) for name, frame in result.items()}),
                    Json({p.name: p.read_text() for p in SQL_FILES})))
        for frame in result.values():
            frame.attrs['database_schema'] = schema
        logger.info('Committed PostgreSQL snapshot %s', schema)
        return result
    finally:
        connection.close()


def read_snapshot(schema, *, include_test=False):
    """Read retained prepared tables without rerunning ingestion or feature SQL."""
    if not isinstance(schema, str) or not schema:
        raise ValueError('Prepared data requires a nonempty database_schema.')
    connection = connect_database()
    try:
        connection.set_session(readonly=True, isolation_level='REPEATABLE READ')
        with connection:
            with connection.cursor() as cursor:
                return {name: read_dataframe(cursor, sql.SQL(
                    'SELECT * FROM {}.{} ORDER BY election, constituency_id, constituency_name'
                ).format(sql.Identifier(schema), sql.Identifier(name + '_data')))
                    for name in (('train', 'test') if include_test else ('train',))}
    finally:
        connection.close()
