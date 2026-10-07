"""Final-model persistence retains preprocessing and ensemble predictions."""
import numpy as np
import mlflow.pyfunc
import pytest
import torch

from election.models import custom_model
from election.models.saved_model import save_model
from tests.test_model_training import model_spec, small_spec, sample


@pytest.mark.parametrize('index', range(8))
def test_saved_model_roundtrip_preserves_predictions(index, tmp_path):
    torch.set_num_threads(1)
    spec = small_spec(model_spec(index))
    data = sample()
    train, test = data[data.election < 1997], data[data.election == 1997]
    context = custom_model.fit_context(
        validation=test if spec['model_id'].startswith('nn') else None)
    fitted = spec['factory']()
    fitted.train(train, spec['configuration'], context)
    inputs = test.drop(columns='winner')
    expected = fitted.predict(inputs)
    path = tmp_path / 'model'
    save_model(fitted, path)
    assert (path / 'MLmodel').exists()
    assert (path / 'code' / 'election' / 'models').is_dir()
    loaded = mlflow.pyfunc.load_model(str(path))
    np.testing.assert_array_equal(loaded.predict(inputs), expected)
