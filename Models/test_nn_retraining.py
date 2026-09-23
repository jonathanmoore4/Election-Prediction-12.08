"""Check learning-rate selection, temporal holdouts, and checkpoint restoration."""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from Models import NN01_model as nn


def sample_data():
    data = pd.DataFrame({column: [0.5] * 24 for column in nn.FEATURE_COLUMNS})
    for column in nn.CATEGORICAL_COLUMNS:
        data[column] = ['a', 'b'] * 12
    data['election'] = [2015] * 8 + [2017] * 8 + [2019] * 8
    data['winner'] = ['lab', 'con'] * 12
    return data


class NeuralNetworkRetrainingTests(unittest.TestCase):
    def test_retraining_searches_rates_and_shifts_holdout(self):
        data = sample_data()
        model = nn.NeuralNetworkModel()
        for latest, fit, winning_rate in [(2017, model.train, 0.1), (2019, model.retrain, 0.3)]:
            supplied = data.loc[data.election <= latest]
            checkpoints = {rate: [] for rate in nn.LEARNING_RATES}
            splits = {}

            def train(training, validation, labels, seed, epochs, learning_rate):
                self.assertEqual(set(validation.election), {latest})
                self.assertEqual(len(validation), 8)
                self.assertEqual(len(training.loc[training.election == latest]), 0)
                self.assertEqual(set(training.index) & set(validation.index), set())
                self.assertEqual(set(training.index) | set(validation.index), set(supplied.index))
                split = tuple(validation.index)
                self.assertEqual(splits.setdefault(seed, split), split)
                self.assertEqual(epochs, nn.MAX_EPOCHS)
                checkpoint = (learning_rate, seed)
                checkpoints[learning_rate].append(checkpoint)
                # Individual losses favour a different rate than ensemble loss.
                loss = 0.8 if learning_rate == winning_rate else 0.2
                return checkpoint, object(), dict(
                    seed=seed, learning_rate=learning_rate, best_epoch=3,
                    stopping_epoch=23, best_validation_loss=loss,
                )

            def probabilities(checkpoint, preprocessor, frame):
                rate, seed = checkpoint
                correct = 0.9 if rate == winning_rate else 0.6
                targets = nn.LabelEncoder().fit(supplied.winner).transform(frame.winner)
                result = np.full((len(frame), 2), 1 - correct)
                result[np.arange(len(frame)), targets] = correct
                return result

            model.hyperparameters['epochs'] = 1
            with patch.object(nn, '_train_network', side_effect=train) as trainer, patch.object(nn, '_probabilities', side_effect=probabilities):
                with contextlib.redirect_stdout(io.StringIO()):
                    fit(supplied)
            self.assertEqual(trainer.call_count, 40)
            self.assertEqual(model.networks, checkpoints[winning_rate])
            self.assertEqual(model.selected_learning_rate, winning_rate)
            self.assertEqual(model.hyperparameters['learning_rate'], winning_rate)
            self.assertTrue(model.learning_rate_summary['runs'].eq(10).all())
            self.assertAlmostEqual(model.learning_rate_summary.loc[winning_rate, 'log_loss'], -np.log(0.9))
        self.assertEqual(len(model.selection_records), 40)
        self.assertEqual(len(model.final_records), 40)
        self.assertEqual(model.selection_learning_rate_summary['log_loss'].idxmin(), 0.1)
        self.assertEqual(model.final_learning_rate_summary['log_loss'].idxmin(), 0.3)

    def test_validation_only_party_is_rejected_before_training(self):
        data = sample_data()
        data.loc[data.election == 2019, 'winner'] = 'new_party'
        with patch.object(nn, '_train_network') as trainer:
            with self.assertRaisesRegex(ValueError, 'absent from weight-update'):
                nn.NeuralNetworkModel().train(data)
            trainer.assert_not_called()

    def test_real_fit_predict_and_retrain(self):
        data = sample_data()
        model = nn.NeuralNetworkModel()
        with patch.object(nn, 'MAX_EPOCHS', 2), patch.object(nn, 'SEEDS', nn.SEEDS[:2]):
            with contextlib.redirect_stdout(io.StringIO()):
                model.train(data.loc[data.election < 2019])
                model.retrain(data)
        self.assertEqual(model.validation_year, 2019)
        self.assertEqual(model.training_elections, [2015, 2017])
        probabilities = model.predict_proba(data)
        self.assertEqual(probabilities.shape, (24, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
        self.assertEqual(model.predict(data).shape, (24,))
        self.assertEqual(model.predict_proba(data.iloc[:0]).shape, (0, 2))

    def test_cross_entropy_checkpoint_and_patience(self):
        data = sample_data()
        training, validation = data.iloc[:16], data.iloc[16:]
        labels = nn.LabelEncoder().fit(data.winner)
        # Tiny decreases must reset patience; then 20 identical losses stop it.
        losses = [1.0, 0.99999] + [0.99999] * 20
        actual_loss = nn.nn.CrossEntropyLoss()

        def criterion(logits, targets):
            self.assertEqual(targets.dtype, torch.long)
            if torch.is_inference_mode_enabled():
                return torch.tensor(losses.pop(0))
            return actual_loss(logits, targets)

        with patch.object(nn.nn, 'CrossEntropyLoss', return_value=criterion):
            network, preprocessor, record = nn._train_network(
                training, validation, labels, nn.SEEDS[0], 100, 0.03,
            )
        self.assertEqual(record['best_epoch'], 2)
        self.assertEqual(record['stopping_epoch'], 22)
        # A deterministic two-epoch run must match the restored checkpoint.
        expected, _, _ = nn._train_network(training, None, labels, nn.SEEDS[0], 2, 0.03)
        for key, value in network.state_dict().items():
            torch.testing.assert_close(value, expected.state_dict()[key], rtol=0, atol=0)
        model = nn.NeuralNetworkModel()
        model.networks = [network]
        model.preprocessors = [preprocessor]
        model.label_encoder = labels
        model.classes_ = labels.classes_
        missing = validation.copy()
        missing.loc[:, nn.NUMERIC_COLUMNS[0]] = np.nan
        probabilities = model.predict_proba(missing)
        self.assertEqual(probabilities.shape, (8, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
