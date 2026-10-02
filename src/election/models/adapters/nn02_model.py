"""NN02: 64/32 validation-selected rate. Ten seeds; evaluator-supplied validation; patience 20.

Uses the shared implementation in nn01_model without changing its configuration.
See analysis/05_nn_first_development/.
"""
import election.models.adapters.nn01_model as nn01_model_module

HIDDEN_SIZES = (64, 32)
LEARNING_RATES = (0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
CLASS_LABELS = ('con', 'lab', 'lib', 'natSW', 'oth')


class NeuralNetworkModel(nn01_model_module.NeuralNetworkModel):
    def __init__(self):
        super().__init__(hidden_sizes=HIDDEN_SIZES, learning_rates=LEARNING_RATES,
                         name='NN02: 64/32 validation-selected rate', class_labels=CLASS_LABELS)
