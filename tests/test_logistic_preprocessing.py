"""Logistic preprocessing retains prediction rows and learns fills in training."""
import unittest
import numpy as np
import pandas as pd
import election.models.adapters.logistic_regression as logistic_regression_module


def sample():
    data = pd.DataFrame({column: np.linspace(0.1, 0.8, 12) for column in logistic_regression_module.FEATURE_COLUMNS})
    data['election'] = [2015]*6 + [2019]*6
    data['winner'] = ['lab', 'con']*6
    data['previous_winner'] = ['con']*12
    data['country/region'] = ['England']*12
    data['incumbent'] = ['con']*12
    return data


class LogisticPreprocessingtests(unittest.TestCase):
    def test_numeric_strings_missing_values_and_unseen_categories(self):
        data = sample()
        data['previous_lib_share'] = data.previous_lib_share.astype(str).astype(object)
        data.loc[0, 'previous_lib_share'] = None
        data['projected_lib_share'] = None
        data.loc[1, 'country/region'] = None
        original = data.copy(deep=True)
        model = logistic_regression_module.LogisticRegressionModel()
        model.train(data)
        future = data.drop(columns='winner').copy()
        future['election'] = 2024
        future.loc[0, logistic_regression_module.NUMERIC_COLUMNS] = np.nan
        future.loc[0, 'previous_winner'] = None
        future.loc[1, 'country/region'] = 'unseen'
        future.loc[2, 'previous_lib_share'] = np.inf
        predictions = model.predict(future)
        self.assertEqual(len(predictions), len(future))
        self.assertTrue(pd.notna(predictions).all())
        transformed = model.pipeline[:-1].transform(future)
        if hasattr(transformed, 'toarray'):
            transformed = transformed.toarray()
        self.assertTrue(np.isfinite(transformed).all())
        pd.testing.assert_frame_equal(data, original)
        model.train(data)
        self.assertEqual(len(model.predict(future)), len(future))


if __name__ == '__main__':
    unittest.main()
