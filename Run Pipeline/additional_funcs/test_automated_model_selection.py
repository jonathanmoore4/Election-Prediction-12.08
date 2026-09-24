"""Check held-out diagnostics without fitting the candidate estimators."""

from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

import automated_model_selection as selection


class ChangedSeatAccuracyTests(unittest.TestCase):
    def run_selection(self, previous_winners, expected_model=0, predictions=None):
        data = pd.DataFrame({
            column: [0.5] * 5
            for column in selection.logistic_regression.FEATURE_COLUMNS
        })
        data["election"] = ["2017", "2019", "2019", "2019", "2019"]
        data["winner"] = ["lab", "con", "lab", "con", "lab"]
        data["previous_winner"] = ["lab", *previous_winners]
        data.loc[4, "lab_polling"] = None
        data.index = [10, 20, 30, 40, 50]
        original = data.copy(deep=True)
        models = [
            Mock(name="overall", predict=Mock(return_value=["con", "lab", "lab"])),
            Mock(name="changed", predict=Mock(return_value=["lab", "con", "con"])),
            Mock(name="third", predict=Mock(return_value=["lab", "con", "lab"])),
            Mock(name="fourth", predict=Mock(return_value=["lab", "con", "lab"])),
            Mock(name="fifth", predict=Mock(return_value=["lab", "con", "lab"])),
            Mock(name="sixth", predict=Mock(return_value=["lab", "con", "lab"])),
        ]
        if predictions is not None:
            for model, values in zip(models, predictions):
                model.predict.return_value = values
        factories = [
            (selection.xgboost_model, "XGBoostModel"),
            (selection.random_forest, "RandomForestModel"),
            (selection.logistic_regression, "LogisticRegressionModel"),
            (selection.NN01_model, "NeuralNetworkModel"),
            (selection.NN02_model, "NeuralNetworkModel"),
            (selection.NN03_model, "NeuralNetworkModel"),
        ]
        with ExitStack() as stack, tempfile.TemporaryDirectory() as directory:
            for index, ((module, factory), model) in enumerate(zip(factories, models)):
                model.name = f"model_{index}"
                stack.enter_context(patch.object(module, factory, return_value=model))
            scores_path = Path(directory) / "scores.csv"
            with redirect_stdout(StringIO()):
                best, name = selection.automated_model_selection(data, scores_path)
            scores = pd.read_csv(scores_path)
        self.assertIs(best, models[expected_model])
        self.assertEqual(name, f"model_{expected_model}")
        best.retrain.assert_called_once_with(data)
        pd.testing.assert_frame_equal(data, original)
        self.assertEqual(scores["evaluation_rows"].tolist(), [3] * 6)
        return best, scores

    def test_changed_subset_scores_and_combined_selection(self):
        best, scores = self.run_selection(["con", "lab", "lab", "con"], expected_model=1)
        self.assertEqual(best.changed_seat_evaluation_rows, 1)
        self.assertEqual(best.initial_changed_seat_accuracies["model_0"], 0.0)
        self.assertEqual(best.initial_changed_seat_accuracies["model_1"], 1.0)
        self.assertAlmostEqual(best.initial_accuracies["model_0"], 2 / 3)
        self.assertEqual(scores["changed_seat_accuracy"].tolist(), [0, 1, 0, 0, 0, 0])
        self.assertEqual(scores["changed_seat_evaluation_rows"].tolist(), [1] * 6)
        self.assertAlmostEqual(scores.loc[0, "selection_score"], 2 / 3)
        self.assertAlmostEqual(scores.loc[1, "selection_score"], 1 + 1 / 3)

    def test_combined_tie_preserves_candidate_order(self):
        self.run_selection(
            ["con", "lab", "lab", "con"],
            predictions=[["con", "lab", "con"]] * 6,
        )

    def test_each_new_neural_network_can_win_selection(self):
        for candidate in [4, 5]:
            predictions = [["lab", "con", "lab"] for _ in range(6)]
            predictions[candidate] = ["con", "lab", "con"]
            self.run_selection(["con", "lab", "lab", "con"], expected_model=candidate, predictions=predictions)

    def test_no_changed_seats_records_unavailable_accuracy(self):
        best, scores = self.run_selection(["con", "lab", "con", None])
        self.assertEqual(best.changed_seat_evaluation_rows, 0)
        self.assertTrue(all(value is None for value in best.initial_changed_seat_accuracies.values()))
        self.assertTrue(scores["changed_seat_accuracy"].isna().all())
        self.assertEqual(scores["selection_score"].tolist(), scores["accuracy"].tolist())


if __name__ == "__main__":
    unittest.main()
