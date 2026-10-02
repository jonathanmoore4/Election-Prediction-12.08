"""Check historical majority fills and training-only numeric medians."""
import unittest
import numpy as np
import pandas as pd
import election.models.missing_data as missing_data_module


class MissingDatatests(unittest.TestCase):
    def test_preceding_election_not_pooled_majority_or_prediction_outcome(self):
        train = pd.DataFrame({
            'election': [2015]*3 + [2019]*3,
            'winner': ['lab', 'lab', 'con', 'con', 'con', 'lab'],
            'previous_winner': ['lib']*6,
            'previous_lib_share': [0.1, 0.2, np.nan, 0.3, 0.4, 0.5],
            'projected_lib_share': [0.2, 0.3, 0.4, 0.5, np.nan, 0.6],
        })
        original = train.copy(deep=True)
        imputer = missing_data_module.ElectionImputer().fit(train)
        future = pd.DataFrame({'election':[2019,2024], 'winner':['oth','oth'],
                               'previous_winner':[np.nan,np.nan],
                               'previous_lib_share':[np.nan,np.nan],
                               'projected_lib_share':[np.nan,np.nan]})
        result = imputer.transform(future)
        self.assertEqual(result.previous_winner.tolist(), ['lab','con'])
        np.testing.assert_allclose(result.previous_lib_share, 0.3)
        np.testing.assert_allclose(result.projected_lib_share, 0.4)
        self.assertTrue(future.previous_winner.isna().all())
        pd.testing.assert_frame_equal(train, original)
        np.testing.assert_array_equal(imputer.transform(train).winner, train.winner)

    def test_sklearn_target_and_earliest_election_fallback(self):
        frame = pd.DataFrame({'election':[2019]*3, 'previous_winner':['lab','lab',None],
                              'previous_lib_share':[np.nan]*3})
        imputer = missing_data_module.ElectionImputer().fit(frame, ['con','con','lab'])
        self.assertEqual(imputer.transform(frame).previous_winner.iloc[-1], 'lab')
        future = frame.iloc[-1:].assign(election=2024)
        result = imputer.transform(future)
        self.assertEqual(result.previous_winner.iloc[0], 'con')
        self.assertEqual(result.previous_lib_share.iloc[0], 0)


if __name__ == '__main__':
    unittest.main()
