"""Train an ensemble of neural networks to predict the winning party.

Shared training implementation for NN01, NN02 and NN03.
Development report: Analysis and model development/05_nn_first_development/.
Training and retraining hold out the entire latest supplied election, search
the configured rates (four for NN01) across ten seeds, and retain the rate with the lowest
ensemble validation log loss. Predictions average the ten best checkpoints'
party probabilities.
Importing this module only defines the classes and functions; it does not train.
"""

from typing import Any, NotRequired, TypedDict
from numpy.typing import NDArray
from Models.custom_model import custom_model

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
from Models.logistic_regression import FEATURE_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


# Use ten reproducible seeds and the notebook's learning-rate candidates.
SEEDS = tuple(range(101223, 101233))
BATCH_SIZE = 64
MAX_EPOCHS = 1000
PATIENCE = 20  # Any strictly lower validation loss resets patience.
LEARNING_RATES = (0.1, 0.2, 0.3, 0.5)
HIDDEN_SIZES = (32, 16)  # Number of neurons in the two hidden layers.


def make_preprocessor() -> ColumnTransformer:
    """Create preprocessing to fit on training rows and reuse for prediction."""
    return ColumnTransformer([
        # Fill missing numbers with training medians, then centre and scale them.
        # Keep even entirely missing columns so the feature layout stays stable.
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), NUMERIC_COLUMNS),
        ("categorical", Pipeline([
            # Treat missing categories as a named category. One-hot encoding makes
            # a separate indicator column for each category seen during fitting;
            # unseen categories at prediction time get all-zero indicators.
            ("impute", SimpleImputer(
                strategy="constant", fill_value="__MISSING__", keep_empty_features=True
            )),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_COLUMNS),
    ])


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


class NeuralNetworkModel(custom_model):
    """Own the selected rate's ten networks, preprocessing, and search diagnostics."""

    def __init__(
        self, *, hidden_sizes: tuple[int, ...] | None = None,
        learning_rates: tuple[float, ...] | None = None,
        name: str = "Neural Network", class_labels: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(name)
        self.hidden_sizes = tuple(HIDDEN_SIZES if hidden_sizes is None else hidden_sizes)
        self.learning_rates = tuple(LEARNING_RATES if learning_rates is None else learning_rates)
        self.class_labels = class_labels
        self.networks: list[NeuralNetwork] = []
        self.preprocessors: list[ColumnTransformer] = []
        self.label_encoder: LabelEncoder | None = None
        self.classes_: NDArray[Any] = np.array([])
        self.selection_records: list[TrainingRecord] = []
        self.final_records: list[TrainingRecord] = []
        self.training_records: list[TrainingRecord] = []
        self.learning_rate_summary = pd.DataFrame()
        self.selection_learning_rate_summary = pd.DataFrame()
        self.final_learning_rate_summary = pd.DataFrame()
        self.selected_learning_rate: float | None = None

    def train(self, data: pd.DataFrame) -> None:
        """Search learning rates and retain the winning rate's ten checkpoints."""
        self._fit(data, selecting=True)

    def retrain(self, data: pd.DataFrame) -> None:
        """Repeat the learning-rate search using the latest supplied election."""
        self._fit(data, selecting=False)

    def predict_proba(self, data: pd.DataFrame) -> NDArray[np.floating[Any]]:
        """Return one row per input and one averaged probability per party."""
        if self.label_encoder is None:
            raise RuntimeError("Call train() before predict().")
        if len(data) == 0:
            return np.empty((0, len(self.classes_)))
        probabilities = []
        # Each network must use its own fitted imputation, scaling and encoding.
        for network, preprocessor in zip(self.networks, self.preprocessors):
            probabilities.append(_probabilities(network, preprocessor, data))
        # Give the selected rate's ten networks equal weight, averaging
        # probabilities rather than votes.
        return np.mean(probabilities, axis=0)

    def predict(self, data: pd.DataFrame) -> NDArray[Any]:
        """Choose the most probable party and convert its index back to its label."""
        encoded = self.predict_proba(data).argmax(axis=1)
        assert self.label_encoder is not None  # predict_proba checks fitted state.
        return self.label_encoder.inverse_transform(encoded)

    def _fit(self, data: pd.DataFrame, *, selecting: bool) -> None:
        """Share fitting mechanics while keeping the two training stages explicit."""
        # Election year controls the validation split; winner is the target.
        # Only FEATURE_COLUMNS are passed to the networks as predictors.
        required = ["election", "winner", *FEATURE_COLUMNS]
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError(f"Data is missing required columns: {missing}")
        if data.empty or data["winner"].isna().any():
            raise ValueError("Neural-network training requires nonempty data with known winners.")
        years = pd.to_numeric(data["election"], errors="raise")
        if years.isna().any():
            raise ValueError("Election years must not be missing.")
        # The latest whole election is used only for checkpoint/rate selection.
        validation = data.loc[years == years.max()]
        training = data.loc[years < years.max()]
        if training.empty:
            raise ValueError("Learning-rate selection needs at least two whole elections.")
        label_encoder = LabelEncoder().fit(training["winner"] if self.class_labels is None else self.class_labels)
        if not set(validation["winner"]).issubset(label_encoder.classes_):
            raise ValueError("Validation includes a party absent from weight-update data.")
        validation_targets = label_encoder.transform(validation["winner"])
        validation_year = int(years.max())

        runs_by_learning_rate = {}
        all_records = []
        ensemble_losses = {}
        for learning_rate in self.learning_rates:
            networks, preprocessors, records = [], [], []
            validation_probabilities = []
            for seed in SEEDS:
                network, preprocessor, record = _train_network(
                    training, validation, label_encoder, seed,
                    MAX_EPOCHS, learning_rate,
                    hidden_sizes=self.hidden_sizes,
                )
                record.update(
                    validation_year=validation_year,
                    train_through=int(years.loc[years < years.max()].max()),
                    hit_epoch_cap=record["stopping_epoch"] == MAX_EPOCHS,
                )
                validation_probabilities.append(_probabilities(network, preprocessor, validation))
                networks.append(network)
                preprocessors.append(preprocessor)
                records.append(record)
                all_records.append(record)
            mean_probabilities = np.mean(validation_probabilities, axis=0)
            winning = mean_probabilities[np.arange(len(validation)), validation_targets]
            ensemble_losses[learning_rate] = float(
                -np.log(np.clip(winning, np.finfo(float).eps, 1)).mean()
            )
            runs_by_learning_rate[learning_rate] = (networks, preprocessors, records)

        summary = pd.DataFrame(all_records).groupby("learning_rate", sort=False).agg(
            mean_best_validation_loss=("best_validation_loss", "mean"),
            std_best_validation_loss=("best_validation_loss", "std"),
            mean_best_epoch=("best_epoch", "mean"),
            runs=("seed", "count"),
        )
        summary["log_loss"] = pd.Series(ensemble_losses)
        # idxmin keeps candidate order when mean losses tie, as in the notebook.
        selected_rate = float(summary["log_loss"].idxmin())
        networks, preprocessors, records = runs_by_learning_rate[selected_rate]
        print(summary.to_string())
        print(f"Selected neural-network learning rate: {selected_rate:g}")

        # Commit only after the entire search succeeds. Preserve the initial
        # search diagnostics when final retraining advances the holdout year.
        if selecting:
            self.selection_records = all_records
            self.selection_learning_rate_summary = summary.copy()
            self.final_records = []
            self.final_learning_rate_summary = pd.DataFrame()
        else:
            self.final_records = all_records
            self.final_learning_rate_summary = summary.copy()
        self.training_records = all_records
        self.learning_rate_summary = summary
        self.selected_learning_rate = selected_rate
        self.validation_year = validation_year
        self.training_elections = sorted(years.loc[years < years.max()].unique().tolist())
        self.hyperparameters.update(
            hidden_sizes=self.hidden_sizes, loss="cross_entropy", optimizer="SGD",
            learning_rate=selected_rate, candidate_rates=self.learning_rates,
            patience=PATIENCE, min_delta=0.0, max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE, seeds=SEEDS,
        )
        self.networks = networks
        self.preprocessors = preprocessors
        self.label_encoder = label_encoder
        self.classes_ = label_encoder.classes_


def _probabilities(network, preprocessor, data):
    """Evaluate a checkpoint with its training-only preprocessing."""
    features = as_features(preprocessor.transform(data[FEATURE_COLUMNS]))
    network.eval()
    with torch.inference_mode():
        return torch.softmax(network(features), dim=1).numpy()


def _train_network(
    training: pd.DataFrame, validation: pd.DataFrame | None,
    label_encoder: LabelEncoder, seed: int, epochs: int, learning_rate: float,
    *, hidden_sizes: tuple[int, ...] | None = None,
) -> tuple[NeuralNetwork, ColumnTransformer, TrainingRecord]:
    """Restore the best checkpoint with validation; otherwise run exactly epochs."""
    preprocessor = make_preprocessor()
    # Learn preprocessing only from training rows to avoid leaking validation data.
    x_train = as_features(preprocessor.fit_transform(training[FEATURE_COLUMNS]))
    # Cross-entropy expects integer class indices and raw network logits.
    y_train = torch.tensor(
        label_encoder.transform(training["winner"]), dtype=torch.long,
    )
    if validation is not None:
        x_validation = as_features(preprocessor.transform(validation[FEATURE_COLUMNS]))
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
            TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True,
            generator=torch.Generator().manual_seed(seed), num_workers=0,
        )
        optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.0)
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
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                # Copy the weights so later updates cannot change this checkpoint.
                best_state = copy.deepcopy(network.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= PATIENCE:
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
