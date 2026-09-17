"""Shared model contract, kept separate to avoid circular imports."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from numpy.typing import NDArray


class custom_model(ABC):
    """A mutable model whose training stages keep the same wrapper instance."""

    def __init__(self, name: str) -> None:
        self.name: str = name
        # Only state needed between training stages must be retained here.
        self.hyperparameters: dict[str, Any] = {}
        # Initial held-out accuracies for every candidate, populated on selection.
        self.initial_accuracies: dict[str, float] = {}
        self.initial_changed_seat_accuracies: dict[str, float | None] = {}
        self.changed_seat_evaluation_rows: int = 0

    @abstractmethod
    def train(self, data: pd.DataFrame) -> None:
        """Fit and store model-specific estimators and preprocessing."""

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> NDArray[Any]:
        """Extract predictors internally and return one party label per row."""

    def retrain(self, data: pd.DataFrame) -> None:
        """Repeat training in place unless a subclass needs a separate procedure."""
        self.train(data)
