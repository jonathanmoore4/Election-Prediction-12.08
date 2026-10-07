"""Loadable final models with the same winner prediction interface for every adapter."""
from importlib.metadata import version
from pathlib import Path

import mlflow.pyfunc


class WinnerModel(mlflow.pyfunc.PythonModel):
    def __init__(self, fitted_model):
        self.fitted_model = fitted_model

    def predict(self, context, model_input, params=None):
        return self.fitted_model.predict(model_input)


def save_model(fitted_model, path, *, model_id=None, run_id=None):
    # Bundle project code so saved preprocessing and model classes remain usable
    # after the working implementation is edited. Pin the fitting environment.
    packages = ('mlflow', 'cloudpickle', 'numpy', 'pandas', 'scikit-learn',
                'scipy', 'torch', 'xgboost', 'optuna', 'psycopg2-binary')
    mlflow.pyfunc.save_model(
        path=str(path), python_model=WinnerModel(fitted_model),
        mlflow_model=mlflow.models.Model(model_id=model_id, run_id=run_id),
        code_paths=[str(Path(__file__).resolve().parents[1])],
        pip_requirements=[f'{package}=={version(package)}' for package in packages])
