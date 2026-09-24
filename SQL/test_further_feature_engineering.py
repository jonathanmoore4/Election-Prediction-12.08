"""Check role features against previous constituency results and vote shares."""
from pathlib import Path
import unittest

import duckdb
import pandas as pd


class FurtherFeaturesTests(unittest.TestCase):
    def test_changes_roles_ties_and_missing_polling(self):
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
        with duckdb.connect() as con:
            con.register('with_polling', pd.DataFrame(rows))
            con.execute(Path(__file__).with_name('4_projected_polling.sql').read_text())
            con.execute(Path(__file__).with_name('5_further_feature_engineering.sql').read_text())
            result = con.sql('SELECT * FROM further_model_data ORDER BY row_id').df()
        self.assertEqual(len(result), len(rows))
        self.assertAlmostEqual(result.loc[0, 'projected_con_share'], .35)
        self.assertAlmostEqual(result.loc[0, 'projected_lab_share'], .45)
        self.assertAlmostEqual(result.loc[0, 'projected_lib_share'], .15)
        self.assertFalse(any(column.startswith('previous_nat_') for column in result))
        self.assertAlmostEqual(result.loc[0, 'con_national_change'], -.1)
        self.assertAlmostEqual(result.loc[0, 'holder_national_swing'], -.1)
        self.assertAlmostEqual(result.loc[0, 'challenger_national_swing'], .1)
        self.assertEqual(result.loc[0, 'challenger_polling'], .4)
        self.assertTrue(pd.isna(result.loc[1, 'challenger_polling']))
        self.assertTrue(pd.isna(result.loc[2, 'holder_polling']))
        self.assertEqual(result.loc[2, 'challenger_polling'], .3)
        self.assertEqual(result.loc[3, 'challenger_polling'], .2)
        self.assertTrue(pd.isna(result.loc[4, 'holder_polling']))
        self.assertTrue(result.oth_national_change.isna().all())
        self.assertNotIn('holder_previous_vote_share', result)


if __name__ == '__main__':
    unittest.main()
