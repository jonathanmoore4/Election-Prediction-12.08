"""NN03: 16 units fixed rate 0.3. Ten seeds; latest whole election held out; patience 20.

Uses the shared implementation in NN01_model without changing its configuration.
See Analysis and model development/05_nn_first_development/.
"""
from Models.NN01_model import NeuralNetworkModel as BaseNeuralNetworkModel

HIDDEN_SIZES = (16,)
LEARNING_RATES = (0.3,)
CLASS_LABELS = ('con', 'lab', 'lib', 'natSW', 'oth')


class NeuralNetworkModel(BaseNeuralNetworkModel):
    def __init__(self):
        super().__init__(hidden_sizes=HIDDEN_SIZES, learning_rates=LEARNING_RATES,
                         name='NN03: 16 units fixed rate 0.3', class_labels=CLASS_LABELS)
