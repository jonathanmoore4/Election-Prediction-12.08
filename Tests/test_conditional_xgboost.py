"""Regression checks for conditional combination, missing inputs and chronology."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from Models import conditional_xgboost_model as conditional


def sample():
    data = pd.DataFrame({c: [0.2] * 30 for c in conditional.FEATURE_COLUMNS})
    data['election'] = np.repeat([2010, 2015, 2017], 10)
    data['previous_winner'] = list(conditional.PARTIES) * 6
    data['winner'] = ['con', 'lab', 'lib', 'oth', 'natSW'] * 6
    data['country/region'] = 'England'
    data['incumbent'] = 'con'
    data.index = np.arange(30) * 2
    return data


class ConditionalTests(unittest.TestCase):
    def test_roles_include_every_party_and_oth_is_last_challenger(self):
        data = sample()
        mapper = conditional.RoleMapper().fit(data)
        _, roles = mapper.transform(data)
        np.testing.assert_array_equal(np.sort(roles, axis=1), np.tile(np.arange(5), (30, 1)))
        np.testing.assert_array_equal(roles[:, 0], np.tile(np.arange(5), 6))
        self.assertTrue((roles[roles[:, 0] != 4, -1] == 4).all())
        self.assertTrue((roles[roles[:, 0] == 4, 1:] == np.arange(4)).all())

    def test_combination_uses_soft_probabilities(self):
        data = sample()
        model = conditional.ConditionalXGBoostModel()
        model.mapper = conditional.RoleMapper().fit(data)
        _, roles = model.mapper.transform(data)
        model.change_model = Mock(classes_=np.array([0, 1]))
        model.change_model.predict_proba.return_value = np.tile([0.7, 0.3], (30, 1))
        model.destination_model = Mock(classes_=np.array([1, 4]))
        model.destination_model.predict_proba.return_value = np.tile([0.8, 0.2], (30, 1))
        result = model.predict_proba(data.drop(columns=['winner']))
        rows = np.arange(30)
        np.testing.assert_allclose(result[rows, roles[:, 0]], 0.7)
        np.testing.assert_allclose(result[rows, roles[:, 1]], 0.24)
        np.testing.assert_allclose(result[rows, roles[:, 4]], 0.06)
        np.testing.assert_allclose(result.sum(axis=1), 1)

    def test_real_fit_missing_prediction_inputs_and_retrain(self):
        data = sample()
        original = data.copy(deep=True)
        model = conditional.ConditionalXGBoostModel()
        fits = []
        original_fit = conditional.RoleMapper.fit

        def tracked_fit(mapper, frame):
            fits.append(sorted(frame.election.unique().tolist()))
            return original_fit(mapper, frame)

        with patch.object(conditional, 'PARAM_GRID', {'max_depth': [2], 'n_estimators': [2]}), patch.object(conditional.RoleMapper, 'fit', tracked_fit):
            model.train(data[data.election < 2017])
            self.assertEqual(fits, [[2010], [2010, 2015]])
            model.retrain(data)
        self.assertEqual(fits[-3:], [[2010], [2010, 2015], [2010, 2015, 2017]])
        predict_data = data.drop(columns=['winner']).copy()
        predict_data.loc[predict_data.index[:2], conditional.FEATURE_COLUMNS] = np.nan
        predict_data.loc[predict_data.index[2], 'country/region'] = 'unseen'
        probabilities = model.predict_proba(predict_data)
        self.assertEqual(probabilities.shape, (30, 5))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        self.assertEqual(len(model.predict(predict_data)), 30)
        self.assertEqual(model.predict(predict_data.iloc[:0]).shape, (0,))
        pd.testing.assert_frame_equal(data, original)

    def test_imputation_uses_training_values_only(self):
        train = sample()
        train['previous_winner'] = 'lib'
        mapper = conditional.RoleMapper().fit(train)
        future = train.iloc[:1].copy()
        future.loc[:, conditional.PREVIOUS_COLUMNS] = np.nan
        future['previous_winner'] = np.nan
        features, roles = mapper.transform(future)
        self.assertEqual(roles[0, 0], 1)
        self.assertEqual(features.incumbent_share.iloc[0], 0.2)

    def test_single_class_stage_is_constant(self):
        data = sample()
        features, _ = conditional.RoleMapper().fit(data).transform(data)
        stage = conditional.Stage().fit(features, np.ones(30), {'max_depth': 2, 'n_estimators': 2})
        np.testing.assert_array_equal(stage.predict(features), np.ones(30))
        np.testing.assert_allclose(stage.predict_proba(features), 1)


if __name__ == '__main__':
    unittest.main()
