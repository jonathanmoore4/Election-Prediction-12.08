"""Train an ensemble of neural networks to predict the winning party.

Both training and retraining hold out half of the latest supplied election
for early stopping in each of ten seeded runs. Retraining shifts the split
forward as newer elections are supplied and retains the best checkpoints.
Predictions average the networks' party probabilities.
Importing this module only defines the classes and functions; it does not train.
"""

from typing import Any, TypedDict
from numpy.typing import NDArray
from Models.custom_model import custom_model

import copy
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Use the same predictors and numeric/categorical groupings as logistic regression.
from Models.logistic_regression import FEATURE_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


# Ten randomly sampled seeds, fixed for reproducible ensemble runs.
SEEDS = tuple(range(111223, 111233))
BATCH_SIZE = 64  # Maximum number of training rows per weight update.
MAX_EPOCHS = 1000  # Maximum training epochs per early-stopping run.
PATIENCE = 50  # Stop after this many epochs without a qualifying improvement.
MIN_DELTA = 0.001  # Minimum decrease in validation loss counted as improvement.
LEARNING_RATE = 0.01  # SGD's step size for adjusting network weights.
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
    """CPU classifier whose logits are converted to probabilities for MAE training."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        layers = []
        # Each linear layer learns weighted combinations of its inputs. ReLU sets
        # negative outputs to zero, letting the network learn nonlinear patterns.
        for width in HIDDEN_SIZES:
            layers.extend([nn.Linear(input_dim, width), nn.ReLU()])
            input_dim = width
        # Output one raw score (logit) per party. Convert these to probabilities
        # with softmax when computing the loss or making predictions.
        layers.append(nn.Linear(input_dim, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Calling network(X) passes the rows through every layer in sequence.
        return self.layers(X)


class TrainingRecord(TypedDict):
    """Diagnostics for one seeded training run."""

    seed: int
    best_epoch: int | None
    stopping_epoch: int
    best_validation_loss: float | None


class NeuralNetworkModel(custom_model):
    """Own ten early-stopped networks, their preprocessing, and diagnostics."""

    def __init__(self) -> None:
        super().__init__("Neural Network")
        self.networks: list[NeuralNetwork] = []
        self.preprocessors: list[ColumnTransformer] = []
        self.label_encoder: LabelEncoder | None = None
        self.classes_: NDArray[Any] = np.array([])
        self.selection_records: list[TrainingRecord] = []
        self.final_records: list[TrainingRecord] = []

    def train(self, data: pd.DataFrame) -> None:
        """Fit candidate networks and retain all ten early-stopping checkpoints."""
        self._fit(data, selecting=True)

    def retrain(self, data: pd.DataFrame) -> None:
        """Fit ten fresh checkpoints, holding out half of the latest supplied election."""
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
            features = as_features(preprocessor.transform(data[FEATURE_COLUMNS]))
            network.eval()
            # Disable gradient tracking for prediction; softmax converts each row
            # of raw scores into party probabilities that sum to one.
            with torch.inference_mode():
                probabilities.append(torch.softmax(network(features), dim=1).numpy())
        # Give all ten networks equal weight, averaging probabilities rather than votes.
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
        # All ensemble members in this call share the same party-to-index mapping.
        label_encoder = LabelEncoder().fit(data["winner"])
        # Keep all earlier elections for training and split only the latest.
        latest = data.loc[years == years.max()]
        historical = data.loc[years < years.max()]
        if historical.empty or len(latest) < 2:
            raise ValueError("Epoch-selection training needs earlier elections and at least two latest-election rows.")
        counts = latest["winner"].value_counts()
        # Stratification preserves party proportions where counts and split
        # sizes allow each party to appear in both halves.
        can_stratify = counts.min() >= 2 and len(latest) // 2 >= len(counts)
        if not can_stratify:
            warnings.warn("Latest-election class counts do not permit stratification.", stacklevel=2)

        networks, preprocessors, records = [], [], []
        for seed in SEEDS:
            # Each seed changes both the latest-election split and network
            # initialisation. Hold out half of the latest election internally.
            latest_train, validation = train_test_split(
                latest, test_size=0.5, random_state=seed,
                stratify=latest["winner"] if can_stratify else None,
            )
            training = pd.concat([historical, latest_train])
            network, preprocessor, record = _train_network(
                training, validation, label_encoder, seed,
                MAX_EPOCHS,
            )
            networks.append(network)
            preprocessors.append(preprocessor)
            records.append(record)

        if selecting:
            self.selection_records = records
            self.final_records = []
        else:
            self.final_records = records
        # Commit the new ensemble to this instance after all ten runs succeed.
        self.networks = networks
        self.preprocessors = preprocessors
        self.label_encoder = label_encoder
        self.classes_ = label_encoder.classes_


def _train_network(
    training: pd.DataFrame, validation: pd.DataFrame | None,
    label_encoder: LabelEncoder, seed: int, epochs: int,
) -> tuple[NeuralNetwork, ColumnTransformer, TrainingRecord]:
    """Restore the best checkpoint with validation; otherwise run exactly epochs."""
    preprocessor = make_preprocessor()
    # Learn preprocessing only from training rows to avoid leaking validation data.
    x_train = as_features(preprocessor.fit_transform(training[FEATURE_COLUMNS]))
    # MAE compares party probabilities with one-hot targets (1 for the winner).
    num_classes = len(label_encoder.classes_)
    y_train = nn.functional.one_hot(
        torch.tensor(label_encoder.transform(training["winner"]), dtype=torch.long),
        num_classes=num_classes,
    ).float()
    if validation is not None:
        x_validation = as_features(preprocessor.transform(validation[FEATURE_COLUMNS]))
        y_validation = nn.functional.one_hot(
            torch.tensor(label_encoder.transform(validation["winner"]), dtype=torch.long),
            num_classes=num_classes,
        ).float()

    # Keep initialisation reproducible without changing the caller's RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        network = NeuralNetwork(x_train.shape[1], len(label_encoder.classes_))
        # Shuffle paired features and labels into batches with a separately seeded
        # generator, making the batch order reproducible for this run.
        loader = DataLoader(
            TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True,
            generator=torch.Generator().manual_seed(seed), num_workers=0,
        )
        optimizer = torch.optim.SGD(network.parameters(), lr=LEARNING_RATE, momentum=0.0)
        # Average absolute probability errors over all rows and parties.
        criterion = nn.L1Loss()
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
                loss = criterion(torch.softmax(network(x_batch), dim=1), y_batch)
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
                    torch.softmax(network(x_validation), dim=1), y_validation
                ).item()
            if not np.isfinite(validation_loss):
                raise RuntimeError(f"Non-finite validation loss for seed {seed}, epoch {epoch}")
            if best_loss - validation_loss >= MIN_DELTA:
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
    record: TrainingRecord = {"seed": seed, "best_epoch": best_epoch if validation is not None else None,
              "stopping_epoch": epoch,
              "best_validation_loss": best_loss if validation is not None else None}
    return network, preprocessor, record
