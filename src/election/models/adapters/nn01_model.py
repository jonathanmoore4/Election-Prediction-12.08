"""Neural ensemble adapter: one rate per fit, explicit validation or refit durations.

Election splitting and rate selection belong to HistoricalEvaluator. Inner fits
restore each seed's best checkpoint; refits train fresh networks on all history.
"""

from typing import Any, NotRequired, TypedDict
from numpy.typing import NDArray
import election.models.custom_model as custom_model_module
import election.models.missing_data as missing_data_module

import copy

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Use the same predictors and numeric/categorical groupings as logistic regression.
import election.models.adapters.logistic_regression as logistic_regression_module


# Use ten reproducible seeds and the notebook's learning-rate candidates.
SEEDS = tuple(range(101223, 101233))
BATCH_SIZE = 64
MAX_EPOCHS = 1000
PATIENCE = 20  # Any strictly lower validation loss resets patience.
LEARNING_RATES = (0.1, 0.2, 0.3, 0.5)
HIDDEN_SIZES = (32, 16)  # Number of neurons in the two hidden layers.


def make_preprocessor() -> Pipeline:
    """Create preprocessing to fit on training rows and reuse for prediction."""
    return Pipeline([("missing_data", missing_data_module.ElectionImputer()), ("features", ColumnTransformer([
        # Fill missing numbers with training medians, then centre and scale them.
        # Keep even entirely missing columns so the feature layout stays stable.
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), logistic_regression_module.NUMERIC_COLUMNS),
        ("categorical", Pipeline([
            # Treat missing categories as a named category. One-hot encoding makes
            # a separate indicator column for each category seen during fitting;
            # unseen categories at prediction time get all-zero indicators.
            ("impute", SimpleImputer(
                strategy="constant", fill_value="__MISSING__", keep_empty_features=True
            )),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), logistic_regression_module.CATEGORICAL_COLUMNS),
    ]))])


def as_features(matrix: NDArray[Any] | sparse.spmatrix) -> torch.Tensor:
    """Convert sklearn's feature matrix into a dense float32 PyTorch tensor."""
    # One-hot encoding can produce sparse matrices; these linear layers need dense input.
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    array = np.asarray(matrix, dtype=np.float32)
    # Fail early if preprocessing leaves NaN or infinite values in the predictors.
    if not np.isfinite(array).all():
        raise ValueError("Preprocessed predictors contain non-finite values.")
    return torch.from_numpy(array)


class NeuralNetwork(nn.Module):
    """CPU classifier trained with cross-entropy on raw logits."""

    def __init__(self, input_dim: int, num_classes: int,
                 hidden_sizes: tuple[int, ...] | None = None) -> None:
        super().__init__()
        layers = []
        # Each linear layer learns weighted combinations of its inputs. ReLU sets
        # negative outputs to zero, letting the network learn nonlinear patterns.
        for width in (HIDDEN_SIZES if hidden_sizes is None else hidden_sizes):
            layers.extend([nn.Linear(input_dim, width), nn.ReLU()])
            input_dim = width
        # Output raw logits for cross-entropy, which applies log-softmax internally.
        # Convert logits to probabilities with softmax only for evaluation.
        layers.append(nn.Linear(input_dim, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Calling network(X) passes the rows through every layer in sequence.
        return self.layers(X)


class TrainingRecord(TypedDict):
    """Diagnostics for one seeded training run."""

    seed: int
    learning_rate: float
    best_epoch: int | None
    stopping_epoch: int
    best_validation_loss: float | None
    validation_year: NotRequired[int]
    train_through: NotRequired[int]
    hit_epoch_cap: NotRequired[bool]
    duration: NotRequired[int]


class NeuralNetworkModel(custom_model_module.custom_model):
    """Own only this fit's ten networks, preprocessing and training records."""

    def __init__(self, *, hidden_sizes=None, learning_rates=None,
                 name='NN01: 32/16 validation-selected rate', class_labels=None):
        super().__init__(name)
        self.hidden_sizes = tuple(HIDDEN_SIZES if hidden_sizes is None else hidden_sizes)
        self.learning_rates = tuple(LEARNING_RATES if learning_rates is None else learning_rates)
        self.class_labels = tuple(class_labels or custom_model_module.PARTIES)
        self.networks = []
        self.preprocessors = []
        self.label_encoder = None

    def train(self, data, configuration=None, fit_context=None):
        context = fit_context or custom_model_module.FitContext(classes=self.class_labels)
        custom_model_module.validate_fit(data, context)
        parameters = dict(hidden_sizes=self.hidden_sizes, learning_rate=self.learning_rates[0],
                          seeds=SEEDS, batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
                          patience=PATIENCE, min_delta=0.0, momentum=0.0, weight_decay=0.0)
        parameters.update(configuration or {})
        seeds = tuple(parameters['seeds'])
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError('Neural seeds must be nonempty and unique.')
        if context.validation is None and set(context.durations) != set(seeds):
            raise ValueError('A neural refit requires a derived duration for every seed.')
        if context.validation is not None and context.durations:
            raise ValueError('Supply validation OR refit durations, not both.')
        encoder = LabelEncoder().fit(list(context.classes))
        networks, preprocessors, runs = [], [], []
        for seed in seeds:
            epochs = (parameters['max_epochs'] if context.validation is not None
                      else context.durations[seed])
            if not isinstance(epochs, (int, np.integer)) or epochs < 1:
                raise ValueError('Training duration must be a positive integer.')
            network, preprocessor, run = _train_network(
                data, context.validation, encoder, seed, epochs, parameters['learning_rate'],
                hidden_sizes=tuple(parameters['hidden_sizes']),
                batch_size=parameters['batch_size'], patience=parameters['patience'],
                min_delta=parameters['min_delta'], momentum=parameters['momentum'],
                weight_decay=parameters['weight_decay'])
            if context.validation is None:
                run['duration'] = epochs
            networks.append(network)
            preprocessors.append(preprocessor)
            runs.append(run)
        self.networks, self.preprocessors = networks, preprocessors
        self.label_encoder = encoder
        self.selected_learning_rate = parameters['learning_rate']
        self.hidden_sizes = tuple(parameters['hidden_sizes'])
        return self._record(data, parameters, context, runs)

    def predict_proba(self, data):
        if self.label_encoder is None:
            raise RuntimeError('Call train() before predicting.')
        if data.empty:
            return np.empty((0, len(self.classes_)))
        return np.mean(list(self.member_probabilities(data).values()), axis=0)

    def member_probabilities(self, data):
        if self.label_encoder is None:
            raise RuntimeError('Call train() before predicting.')
        if data.empty:
            return {run['seed']: np.empty((0, len(self.classes_))) for run in self.training_records}
        return {
            run['seed']: self._align_probabilities(
                _probabilities(network, preprocessor, data), self.label_encoder.classes_)
            for run, network, preprocessor in zip(self.training_records, self.networks, self.preprocessors)
        }


def _probabilities(network, preprocessor, data):
    """Evaluate a checkpoint with its training-only preprocessing."""
    features = as_features(preprocessor.transform(data))
    network.eval()
    with torch.inference_mode():
        return torch.softmax(network(features), dim=1).numpy()


def _train_network(
    training: pd.DataFrame, validation: pd.DataFrame | None,
    label_encoder: LabelEncoder, seed: int, epochs: int, learning_rate: float,
    *, hidden_sizes: tuple[int, ...] | None = None,
    batch_size=64, patience=20, min_delta=0.0, momentum=0.0, weight_decay=0.0,
) -> tuple[NeuralNetwork, Pipeline, TrainingRecord]:
    """Restore the best checkpoint with validation; otherwise run exactly epochs."""
    preprocessor = make_preprocessor()
    # Learn preprocessing only from training rows to avoid leaking validation data.
    x_train = as_features(preprocessor.fit_transform(training))
    # Cross-entropy expects integer class indices and raw network logits.
    y_train = torch.tensor(
        label_encoder.transform(training["winner"]), dtype=torch.long,
    )
    if validation is not None:
        x_validation = as_features(preprocessor.transform(validation))
        y_validation = torch.tensor(
            label_encoder.transform(validation["winner"]), dtype=torch.long,
        )

    # Keep initialisation reproducible without changing the caller's RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        network = NeuralNetwork(x_train.shape[1], len(label_encoder.classes_), hidden_sizes)
        # Shuffle paired features and labels into batches with a separately seeded
        # generator, making the batch order reproducible for this run.
        loader = DataLoader(
            TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed), num_workers=0,
        )
        optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        # Average negative log probability of the winning class over rows.
        criterion = nn.CrossEntropyLoss()
        best_loss = float("inf")
        best_epoch = 0
        best_state = None
        stale_epochs = 0
        for epoch in range(1, epochs + 1):
            network.train()
            for x_batch, y_batch in loader:
                # Clear old gradients, measure this batch's error, differentiate
                # it with respect to the weights, then apply an SGD update.
                optimizer.zero_grad()
                loss = criterion(network(x_batch), y_batch)
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"Non-finite training loss for seed {seed}, epoch {epoch}")
                loss.backward()
                optimizer.step()
            if validation is None:
                # Refitting uses a fixed duration, with no early-stopping check.
                continue
            # Evaluate held-out rows after each epoch without updating the weights.
            network.eval()
            with torch.inference_mode():
                validation_loss = criterion(
                    network(x_validation), y_validation
                ).item()
            if not np.isfinite(validation_loss):
                raise RuntimeError(f"Non-finite validation loss for seed {seed}, epoch {epoch}")
            if validation_loss < best_loss - min_delta:
                best_loss = validation_loss
                best_epoch = epoch
                # Copy the weights so later updates cannot change this checkpoint.
                best_state = copy.deepcopy(network.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= patience:
                break
        if validation is not None:
            # Use the best qualifying checkpoint, which can precede the stopping epoch.
            network.load_state_dict(best_state)
        network.eval()
    # Keep diagnostics for inspecting duration selection and subsequent refits.
    record: TrainingRecord = {"seed": seed, "learning_rate": learning_rate, "best_epoch": best_epoch if validation is not None else None,
              "stopping_epoch": epoch,
              "best_validation_loss": best_loss if validation is not None else None}
    return network, preprocessor, record
