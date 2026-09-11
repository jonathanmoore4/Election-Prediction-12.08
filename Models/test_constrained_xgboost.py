"""Focused checks for leakage isolation, new-election intervals, and assignment."""

import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import t

from Models.constrained_xgboost import (
    PARTIES, POLL_COLUMNS, assign_constituencies, load_inputs, polling_seat_ranges,
)


class ConstrainedForecastTests(unittest.TestCase):
    def test_binding_constraints_match_exhaustive_optimum_with_zero_probabilities(self):
        probabilities = pd.DataFrame([
            [.7, .2, .1, 0, 0], [.6, .3, 0, .1, 0], [.5, .1, .1, .1, .2],
        ], columns=PARTIES)
        ranges = pd.DataFrame({'lower_seats': [0, 1, 1], 'upper_seats': [1, 1, 1]},
                              index=list(POLL_COLUMNS))
        winners, solution = assign_constituencies(probabilities, ranges)
        costs = -np.log(np.clip(probabilities.to_numpy(), 1e-15, 1))
        feasible_costs = []
        for assignment in itertools.product(range(5), repeat=3):
            counts = np.bincount(assignment, minlength=5)
            if all(ranges.iloc[j].lower_seats <= counts[j] <= ranges.iloc[j].upper_seats for j in range(3)):
                feasible_costs.append(sum(costs[i, party] for i, party in enumerate(assignment)))
        self.assertAlmostEqual(solution.fun, min(feasible_costs))
        self.assertEqual(sum(winners == 'Conservative'), 1)
        self.assertEqual(sum(winners == 'Liberal Democrat'), 1)
        self.assertFalse(np.array_equal(winners, probabilities.idxmax(axis=1)))

    def test_other_and_natsw_remain_available(self):
        probabilities = pd.DataFrame([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], columns=PARTIES)
        ranges = pd.DataFrame({'lower_seats': [0, 0, 0], 'upper_seats': [0, 0, 0]}, index=list(POLL_COLUMNS))
        winners, _ = assign_constituencies(probabilities, ranges)
        self.assertEqual(winners.tolist(), ['Other', 'natSW'])
        ranges['lower_seats'] = 1
        ranges['upper_seats'] = 2
        with self.assertRaisesRegex(ValueError, 'infeasible'):
            assign_constituencies(probabilities, ranges)

    def test_interval_includes_new_election_error(self):
        rows = []
        x = np.array([.1, .2, .3, .4])
        y = np.array([1, 3, 2, 5])
        for year, poll, seats in zip([2001, 2005, 2010, 2017], x, y):
            for i in range(10):
                rows.append(dict(election=year, Labour=poll, Conservative=poll, LD=poll,
                                 winner='lab' if i < seats else 'con'))
        historical = pd.DataFrame(rows)
        forecast = pd.DataFrame(dict(election=[2019] * 10, Labour=[.25] * 10,
                                     Conservative=[.25] * 10, LD=[.25] * 10))
        ranges, _, _ = polling_seat_ranges(historical, forecast)
        design = np.column_stack([np.ones(4), x])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        variance = np.sum((y - design @ beta) ** 2) / 2
        expected_half_width = t.ppf(.975, 2) * np.sqrt(variance * 1.25)
        row = ranges.loc['Labour']
        self.assertAlmostEqual(row.prediction_upper_95 - row.predicted_seats, expected_half_width)
        self.assertGreater(expected_half_width, t.ppf(.975, 2) * np.sqrt(variance * .25))

    def test_2019_outcome_values_cannot_enter_inputs(self):
        root = Path(__file__).resolve().parents[1]
        historical, forecast = load_inputs(root / 'TEST_TRAIN/train.csv')
        # Fabricate two incompatible 2019 labels; never inspect actual outcomes.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.csv'
            for label in ['UNUSABLE_LABEL', None]:
                fabricated = forecast.assign(winner=label, con_share=999, majority_proportion=-999)
                pd.concat([historical, fabricated], ignore_index=True).to_csv(path, index=False)
                loaded_history, loaded_forecast = load_inputs(path)
                pd.testing.assert_frame_equal(loaded_history, historical, check_exact=False)
                pd.testing.assert_frame_equal(loaded_forecast, forecast, check_exact=False)
        self.assertNotIn('winner', forecast)
        self.assertTrue(historical.election.le(2017).all())


if __name__ == '__main__':
    unittest.main()
