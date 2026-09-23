"""NN02: 64/32 validation-selected rate. Ten seeds; latest whole election held out; patience 20.

Uses the shared implementation in NN01_model without changing its configuration.
See Analysis and model development/05_nn_first_development/.
"""
from Models.NN01_model import NeuralNetworkModel as BaseNeuralNetworkModel

HIDDEN_SIZES = (64, 32)
LEARNING_RATES = (0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
CLASS_LABELS = ('con', 'lab', 'lib', 'natSW', 'oth')


class NeuralNetworkModel(BaseNeuralNetworkModel):
    def __init__(self):
        super().__init__(hidden_sizes=HIDDEN_SIZES, learning_rates=LEARNING_RATES,
                         name='NN02: 64/32 validation-selected rate', class_labels=CLASS_LABELS)
