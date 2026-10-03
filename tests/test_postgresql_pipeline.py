"""Exercise bulk loading, feature SQL, snapshot isolation and rollback on PostgreSQL."""
import os

import pandas as pd
import pytest
from psycopg2 import sql

from election.sql import apply_sql_queries as database


def cleaned_inputs():
    base = dict(constituency_name='Example', **{'country/region': 'England'},
                winning_party_vote_share=.45, second_party_vote_share=.35,
                winner='con', constituency_id='E001', boundary_set='old',
                election_type='actual', con_share=.45, lib_share=.1, lab_share=.35,
                natSW_share=0., con_national_vote_share=.4, lab_national_vote_share=.3,
                lib_national_vote_share=.15, natSW_national_vote_share=.05,
                oth_national_vote_share=.1)
    frames = {
        'historical': pd.DataFrame([dict(base, election=y, previous_election=p)
            for y, p in [('1983', '1979'), ('1987', '1983'), ('2019', '1987')]]),
        'results_2024': pd.DataFrame([dict(base, election='2024', previous_election='2019_notional')]),
        'polling': pd.DataFrame([dict(Date=y, con_polling=.3, lab_polling=.4,
                                      lib_polling=.2, incumbent='con')
                                for y in ['1987', '2019', '2024']]),
    }
    for year in ['1992', '2001', '2005', '2019']:
        frames['notional_' + year] = pd.DataFrame([
            dict(base, election=year + '_notional', previous_election='1987')])
    return frames


def test_bulk_load_preserves_nulls_strings_and_identifiers(postgres_cursor):
    frame = pd.DataFrame({'MixedCase': ['', None, 'NULL', 'comma,quote"\nline'],
                          'float_value': [1.2, float('nan'), 3.4, float('nan')],
                          'number': pd.Series([1, None, 3, 4], dtype='Int64'),
                          'flag': pd.Series([True, False, None, True], dtype='boolean')})
    database.load_dataframe(postgres_cursor, 'input_quoted', frame)
    result = database.read_dataframe(postgres_cursor, 'SELECT * FROM input_quoted')
    assert result.MixedCase.iloc[0] == ''
    assert pd.isna(result.MixedCase.iloc[1])
    assert result.MixedCase.iloc[2:].tolist() == ['NULL', 'comma,quote"\nline']
    assert result.float_value.isna().tolist() == [False, True, False, True]
    assert result.number.iloc[0] == 1
    assert pd.isna(result.number.iloc[1])
    assert result.flag.tolist() == [True, False, None, True]


def test_missing_database_configuration_is_actionable(monkeypatch, tmp_path):
    monkeypatch.delenv('ELECTION_DATABASE_URL', raising=False)
    monkeypatch.setattr(database.paths, 'project_root', lambda: tmp_path)
    with pytest.raises(ValueError, match='ELECTION_DATABASE_URL'):
        database.connect_database()


@pytest.mark.parametrize('explicit,environment,expected', [
    (None, None, 'postgresql://local/election'),
    (None, 'postgresql://environment/election', 'postgresql://environment/election'),
    ('postgresql://explicit/election', 'postgresql://environment/election',
     'postgresql://explicit/election'),
])
def test_database_configuration_precedence(monkeypatch, tmp_path, explicit, environment, expected):
    monkeypatch.delenv('ELECTION_DATABASE_URL', raising=False)
    if environment:
        monkeypatch.setenv('ELECTION_DATABASE_URL', environment)
    monkeypatch.setattr(database.paths, 'project_root', lambda: tmp_path)
    (tmp_path / '.env.election').write_text(
        "# Local setup\nexport ELECTION_DATABASE_URL='postgresql://local/election'\n")
    calls = []
    sentinel = object()

    def connect(url, **kwargs):
        calls.append(url)
        return sentinel

    monkeypatch.setattr(database.psycopg2, 'connect', connect)
    monkeypatch.chdir(tmp_path.parent)
    assert database.connect_database(explicit) is sentinel
    assert calls == [expected]
    assert os.environ.get('ELECTION_DATABASE_URL') == environment


def test_snapshots_are_persistent_isolated_and_failed_runs_roll_back(monkeypatch, tmp_path):
    if not os.environ.get('ELECTION_DATABASE_URL'):
        pytest.skip('Set ELECTION_DATABASE_URL; see docs/postgresql.md')
    connection = database.connect_database()
    schemas = []
    try:
        original = cleaned_inputs()
        first = database.apply_sql_queries(original)
        schemas.append(first['train'].attrs['database_schema'])
        assert first['train'].election.tolist() == ['1987', '2019']
        assert first['test'].election.tolist() == ['2024']
        assert first['test'].previous_con_share.iloc[0] == .45
        assert first['test'].projected_con_share.iloc[0] == pytest.approx(.35)
        assert first['test'].natSW_national_change.isna().all()
        assert first['test'].natSW_national_change.dtype.kind == 'f'
        changed = cleaned_inputs()
        changed['polling']['con_polling'] = .5
        second = database.apply_sql_queries(changed)
        schemas.append(second['train'].attrs['database_schema'])
        assert schemas[0] != schemas[1]
        assert second['test'].projected_con_share.iloc[0] == pytest.approx(.55)
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL('SELECT projected_con_share FROM {}.test_data').format(sql.Identifier(schemas[0])))
            assert cursor.fetchone()[0] == pytest.approx(.35)
            cursor.execute(sql.SQL('SELECT input_rows, output_rows FROM {}.snapshot_info').format(sql.Identifier(schemas[0])))
            inputs, outputs = cursor.fetchone()
            assert inputs['historical'] == 3
            assert outputs == {'train': 2, 'test': 1}
            cursor.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE 'run_%'")
            before = set(cursor.fetchall())
        bad_sql = tmp_path / 'invalid.sql'
        bad_sql.write_text('SELECT no_such_column FROM historical')
        monkeypatch.setattr(database, 'SQL_FILES', [*database.SQL_FILES, bad_sql])
        import psycopg2
        with pytest.raises(psycopg2.errors.UndefinedColumn):
            database.apply_sql_queries(original)
        with connection.cursor() as cursor:
            cursor.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE 'run_%'")
            assert set(cursor.fetchall()) == before
    finally:
        connection.rollback()
        with connection:
            with connection.cursor() as cursor:
                for schema in schemas:
                    cursor.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
        connection.close()
