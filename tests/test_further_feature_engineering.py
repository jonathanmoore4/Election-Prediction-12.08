"""Check role features against previous constituency results and vote shares."""
from pathlib import Path

from election.sql.apply_sql_queries import load_dataframe, read_dataframe
import pandas as pd
import pytest


class TestFurtherFeatures:
    def test_changes_roles_ties_and_missing_polling(self, postgres_cursor):
        base = dict(previous_winner='con', previous_con_share=.45,
                    previous_lib_share=.1, previous_lab_share=.35,
                    previous_natSW_share=0.,
                    previous_winning_party_last_election_vote_share=.45,
                    previous_second_party_last_election_vote_share=.35,
                    con_polling=.3, lab_polling=.4, lib_polling=.2,
                    previous_con_national_vote_share=.4,
                    previous_lab_national_vote_share=.3,
                    previous_lib_national_vote_share=.15)
        rows = [base, {**base, 'previous_second_party_last_election_vote_share': .4},
                {**base, 'previous_winner': 'natSW', 'previous_natSW_share': .5,
                 'previous_second_party_last_election_vote_share': .45},
                {**base, 'previous_lib_share': .45,
                 'previous_second_party_last_election_vote_share': .45},
                {**base, 'previous_winner': None}]
        rows = [dict(row, row_id=i) for i, row in enumerate(rows)]
        con = postgres_cursor
        load_dataframe(con, 'with_polling', pd.DataFrame(rows))
        con.execute((Path(__file__).resolve().parents[1] / 'src/election/sql' / '4_projected_polling.sql').read_text())
        con.execute((Path(__file__).resolve().parents[1] / 'src/election/sql' / '5_further_feature_engineering.sql').read_text())
        result = read_dataframe(con, 'SELECT * FROM further_model_data ORDER BY row_id')
        assert len(result) == len(rows)
        assert result.loc[0, 'projected_con_share'] == pytest.approx(.35)
        assert result.loc[0, 'projected_lab_share'] == pytest.approx(.45)
        assert result.loc[0, 'projected_lib_share'] == pytest.approx(.15)
        assert not any(column.startswith('previous_nat_') for column in result)
        assert result.loc[0, 'con_national_change'] == pytest.approx(-.1)
        assert result.loc[0, 'holder_national_swing'] == pytest.approx(-.1)
        assert result.loc[0, 'challenger_national_swing'] == pytest.approx(.1)
        assert result.loc[0, 'challenger_polling'] == .4
        assert pd.isna(result.loc[1, 'challenger_polling'])
        assert pd.isna(result.loc[2, 'holder_polling'])
        assert result.loc[2, 'challenger_polling'] == .3
        assert result.loc[3, 'challenger_polling'] == .2
        assert pd.isna(result.loc[4, 'holder_polling'])
        assert result.oth_national_change.isna().all()
        assert 'holder_previous_vote_share' not in result
