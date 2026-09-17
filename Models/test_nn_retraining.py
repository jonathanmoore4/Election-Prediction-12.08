"""Check temporal holdouts across candidate training and final retraining."""
import unittest
from unittest.mock import patch

import pandas as pd

from Models import NN01_model as nn


class NeuralNetworkRetrainingTests(unittest.TestCase):
    def test_retraining_shifts_holdout_and_retains_checkpoints(self):
        data = pd.DataFrame({column: [0.5] * 24 for column in nn.FEATURE_COLUMNS})
        data['election'] = [2015] * 8 + [2017] * 8 + [2019] * 8
        data['winner'] = ['lab', 'con'] * 12
        model = nn.NeuralNetworkModel()
        for latest, fit in [(2017, model.train), (2019, model.retrain)]:
            supplied = data.loc[data.election <= latest]
            checkpoints = []

            def train(training, validation, labels, seed, epochs):
                self.assertEqual(set(validation.election), {latest})
                self.assertEqual(len(validation), 4)
                self.assertEqual(len(training.loc[training.election == latest]), 4)
                self.assertEqual(set(training.index) & set(validation.index), set())
                self.assertEqual(set(training.index) | set(validation.index), set(supplied.index))
                self.assertEqual(epochs, nn.MAX_EPOCHS)
                checkpoint = object()
                checkpoints.append(checkpoint)
                return checkpoint, object(), dict(seed=seed, best_epoch=3,
                                                  stopping_epoch=23, best_validation_loss=0.1)

            # A stale epoch value must not control the final training duration.
            model.hyperparameters['epochs'] = 1
            with patch.object(nn, '_train_network', side_effect=train) as trainer:
                fit(supplied)
            self.assertEqual(trainer.call_count, 10)
            self.assertEqual(model.networks, checkpoints)
        self.assertEqual(len(model.selection_records), 10)
        self.assertEqual(len(model.final_records), 10)
        self.assertTrue(all(r['best_epoch'] == 3 for r in model.final_records))


if __name__ == '__main__':
    unittest.main()
