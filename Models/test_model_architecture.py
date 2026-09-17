"""Small regression checks for model lifecycle and selection orchestration.

Run from the project root with: python -m unittest Models.test_model_architecture
"""

import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from Models import NN01_model, logistic_regression, random_forest, xgboost_model
from Models.custom_model import custom_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Run Pipeline" / "additional_funcs"))
selector = importlib.import_module("automated_model_selection")


def sample_data() -> pd.DataFrame:
    """Three elections with enough observations of each party for splitting."""
    rows = 36
    data = pd.DataFrame({column: np.linspace(0, 1, rows)
                         for column in logistic_regression.NUMERIC_COLUMNS})
    for column in logistic_regression.CATEGORICAL_COLUMNS:
        data[column] = ["a", "b"] * (rows // 2)
    data["winner"] = ["a", "b"] * (rows // 2)
    data["election"] = np.repeat([2015, 2017, 2019], 12)
    return data


class ModelLifecycleTests(unittest.TestCase):
    def test_pipeline_models_retrain_in_place(self) -> None:
        data = sample_data()
        # Exercise the real LR/RF fitting routines. XGBoost's unchanged grid
        # search is checked separately through its wrapper's delegation.
        for cls in (logistic_regression.LogisticRegressionModel,
                    random_forest.RandomForestModel):
            with self.subTest(model=cls.__name__):
                model = cls()
                with self.assertRaises(RuntimeError):
                    model.predict(data)
                self.assertIsNone(model.train(data))
                previous = model.pipeline
                self.assertEqual(model.predict(data).shape, (len(data),))
                self.assertIsNone(model.retrain(data))
                self.assertIsNot(model.pipeline, previous)
                self.assertEqual(model.hyperparameters, {})

    def test_xgboost_repeats_original_trainer(self) -> None:
        data = sample_data()
        model = xgboost_model.XGBoostModel()
        with patch.object(xgboost_model, "train_xgboost") as train:
            self.assertIsNone(model.train(data))
            self.assertIsNone(model.retrain(data))
            self.assertEqual(train.call_count, 2)
            model.predict(data)
            pd.testing.assert_frame_equal(
                train.return_value.predict.call_args.args[0],
                data[xgboost_model.FEATURE_COLUMNS],
            )

    def test_neural_network_ensemble_lifecycle(self) -> None:
        data = sample_data()
        model = NN01_model.NeuralNetworkModel()
        with self.assertRaisesRegex(RuntimeError, "train"):
            model.retrain(data)
        original_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            # Reduce runtime only in this test; retain all ten production seeds.
            with patch.object(NN01_model, "MAX_EPOCHS", 2), patch.object(
                NN01_model, "_train_network", wraps=NN01_model._train_network
            ) as train:
                self.assertIsNone(model.train(data[data.election < 2019]))
                self.assertEqual(len(model.networks), 10)
                self.assertEqual(len({id(p) for p in model.preprocessors}), 10)
                for call in train.call_args_list:
                    training, validation = call.args[:2]
                    self.assertEqual(len(training), 18)
                    self.assertEqual(len(validation), 6)
                    self.assertTrue(set(training.index).isdisjoint(validation.index))
                    self.assertTrue((validation.election == 2017).all())
                epochs = int(np.ceil(np.median([
                    r["best_epoch"] for r in model.selection_records
                ])))
                self.assertEqual(model.hyperparameters["epochs"], epochs)
                probabilities = model.predict_proba(data)
                np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
                self.assertEqual(model.predict(data).shape, (len(data),))
                previous = model.networks[:]
                train.reset_mock()
                self.assertIsNone(model.retrain(data))
                self.assertEqual(len(model.networks), 10)
                self.assertTrue(all(a is not b for a, b in zip(previous, model.networks)))
                for call in train.call_args_list:
                    self.assertIs(call.args[0], data)
                    self.assertIsNone(call.args[1])
                    self.assertEqual(call.args[4], epochs)
                self.assertTrue(all(r["stopping_epoch"] == epochs for r in model.final_records))
                self.assertEqual(len(model.selection_records), 10)
                self.assertEqual(model.predict(data).shape, (len(data),))
        finally:
            torch.set_num_threads(original_threads)

    def test_selector_retrains_same_winner_and_preserves_ties(self) -> None:
        data = sample_data()

        class Candidate(custom_model):
            def train(self, frame: pd.DataFrame) -> None:
                self.training = frame

            def predict(self, frame: pd.DataFrame) -> np.ndarray:
                self.validation = frame
                return frame.winner.to_numpy()

            def retrain(self, frame: pd.DataFrame) -> None:
                self.final = frame

        candidates = [Candidate(str(i)) for i in range(4)]
        with patch.object(xgboost_model, "XGBoostModel", return_value=candidates[0]), \
             patch.object(random_forest, "RandomForestModel", return_value=candidates[1]), \
             patch.object(logistic_regression, "LogisticRegressionModel", return_value=candidates[2]), \
             patch.object(NN01_model, "NeuralNetworkModel", return_value=candidates[3]):
            winner, name = selector.automated_model_selection(data)
        self.assertIs(winner, candidates[0])
        self.assertEqual(name, "0")
        self.assertIs(winner.final, data)
        for model in candidates:
            self.assertTrue((model.training.election < 2019).all())
            self.assertTrue((model.validation.election == 2019).all())
        self.assertTrue(all(not hasattr(m, "final") for m in candidates[1:]))


if __name__ == "__main__":
    unittest.main()
