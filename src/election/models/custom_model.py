"""Common contract: one model instance represents one fit, never a search."""
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

PARTIES = ('con', 'lab', 'lib', 'natSW', 'oth')


@dataclass
class FitContext:
    """Only inner fits receive validation outcomes; refits receive durations."""
    validation: pd.DataFrame | None = None
    classes: tuple[str, ...] = PARTIES
    durations: dict[int, int] = field(default_factory=dict)
    # Scoped to one candidate's tuning call and one training/validation split.
    # Adapters may reuse internal stages, but must not mutate cached fitted state.
    cache: dict = field(default_factory=dict, repr=False)


@dataclass
class FitRecord:
    training_years: tuple[int, ...]
    configuration: dict[str, Any]
    validation_year: int | None = None
    runs: list[dict[str, Any]] = field(default_factory=list)


def validate_fit(data, context):
    if data.empty or data['winner'].isna().any():
        raise ValueError('Fitting requires nonempty labelled training data.')
    years = pd.to_numeric(data.election, errors='raise')
    if not np.isfinite(years).all() or not (years == years.astype(int)).all():
        raise ValueError('Election years must be finite integers.')
    unknown = set(data.winner) - set(context.classes)
    if unknown:
        raise ValueError(f'Unknown training party classes: {sorted(unknown)}')
    validation_year = None
    if context.validation is not None:
        validation = context.validation
        vy = pd.to_numeric(validation.election, errors='raise')
        if validation.empty or vy.nunique() != 1 or not np.isfinite(vy).all():
            raise ValueError('Validation must contain one whole election.')
        validation_year = int(vy.iloc[0])
        if years.max() >= validation_year:
            raise ValueError('All training elections must precede validation.')
        if validation.winner.isna().any() or set(validation.winner) - set(context.classes):
            raise ValueError('Validation requires known labels in the declared classes.')
    return tuple(int(y) for y in sorted(years.unique())), validation_year


class custom_model(ABC):
    def __init__(self, name: str):
        self.name = name
        self.hyperparameters: dict[str, Any] = {}
        self.classes_ = np.asarray(PARTIES)
        self.training_records: list[dict[str, Any]] = []
        self.fit_record: FitRecord | None = None

    @abstractmethod
    def train(self, data, configuration=None, fit_context=None) -> FitRecord:
        """Fit one supplied configuration and store fitted state on this object."""

    @abstractmethod
    def predict_proba(self, data) -> np.ndarray:
        """Return probabilities with columns in self.classes_ order."""

    def predict(self, data) -> np.ndarray:
        return self.classes_[self.predict_proba(data).argmax(axis=1)]

    def member_probabilities(self, data):
        """Optional ensemble diagnostics, keyed by seed; ordinary models return none."""
        return {}

    def _record(self, data, configuration, context, runs=None):
        years, validation_year = validate_fit(data, context)
        self.hyperparameters = deepcopy(configuration)
        self.classes_ = np.asarray(context.classes)
        self.training_records = deepcopy(runs or [])
        self.fit_record = FitRecord(years, deepcopy(configuration), validation_year,
                                    deepcopy(self.training_records))
        return deepcopy(self.fit_record)

    def _align_probabilities(self, probabilities, fitted_classes):
        result = np.zeros((len(probabilities), len(self.classes_)), dtype=float)
        for index, party in enumerate(fitted_classes):
            matches = np.flatnonzero(self.classes_ == party)
            if not len(matches):
                raise ValueError(f'Undeclared fitted class: {party}')
            result[:, matches[0]] = probabilities[:, index]
        return result

    def retrain(self, data, *, spec=None):
        """Compatibility wrapper: retune on new history and return a NEW object.

        Callers must assign the return value. The historical object is unchanged.
        New code should call HistoricalEvaluator.tune_and_refit directly.
        """
        import election.models.evaluation as evaluation_module
        import election.models.candidates as candidates_module
        if spec is None:
            spec = next((s for s in candidates_module.default_candidates() if s.name == self.name), None)
        if spec is None:
            raise ValueError('Supply the CandidateSpec for this model when retraining.')
        return evaluation_module.HistoricalEvaluator([spec]).tune_and_refit(spec, data)
