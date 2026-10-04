"""NN03: 16 units fixed rate 0.3. Ten seeds; evaluator-supplied validation; patience 20.

Uses the shared implementation in nn01_model without changing its configuration.
See analysis/05_nn_first_development/.
"""
import election.models.adapters.nn01_model as nn01_model_module

HIDDEN_SIZES = (16,)
LEARNING_RATES = (0.3,)
CLASS_LABELS = ('con', 'lab', 'lib', 'natSW', 'oth')


class NeuralNetworkModel(nn01_model_module.NeuralNetworkModel):
    def __init__(self):
        super().__init__(hidden_sizes=HIDDEN_SIZES, learning_rates=LEARNING_RATES,
                         name='NN03: 16 units fixed rate 0.3', class_labels=CLASS_LABELS)


nn03hyperparameters = {'learning_rate': list(LEARNING_RATES)}
TRAINING_METADATA = nn01_model_module.training_metadata('nn03', HIDDEN_SIZES)


def NN03(train_data, test_data, hyperparameters, *, fit_records=None,
          cache=None, return_details=False, output_dir=None):
    """Preserve inner checkpoint selection and full-history median-duration refits."""
    return nn01_model_module.evaluate_neural(
        NeuralNetworkModel, TRAINING_METADATA, train_data, test_data, hyperparameters,
        fit_records=fit_records, cache=cache,
        return_details=return_details, output_dir=output_dir)
