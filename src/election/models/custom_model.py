"""Common contract: one model instance represents one fit, never a search."""
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import pandas as pd

from election.models.config import PARTIES


def fit_context(*, validation=None, classes=PARTIES, durations=None, cache=None):
    """Small transient fit settings; never an experiment/result object."""
    return dict(validation=validation, classes=tuple(classes),
                durations={} if durations is None else durations,
                cache={} if cache is None else cache)


def validate_fit(data, context):
    if data.empty or data['winner'].isna().any():
        raise ValueError('Fitting requires nonempty labelled training data.')
    years = pd.to_numeric(data.election, errors='raise')
    if not np.isfinite(years).all() or not (years == years.astype(int)).all():
        raise ValueError('Election years must be finite integers.')
    unknown = set(data.winner) - set(context['classes'])
    if unknown:
        raise ValueError(f'Unknown training party classes: {sorted(unknown)}')
    validation_year = None
    if context['validation'] is not None:
        validation = context['validation']
        vy = pd.to_numeric(validation.election, errors='raise')
        if validation.empty or vy.nunique() != 1 or not np.isfinite(vy).all():
            raise ValueError('Validation must contain one whole election.')
        validation_year = int(vy.iloc[0])
        if years.max() >= validation_year:
            raise ValueError('All training elections must precede validation.')
        if validation.winner.isna().any() or set(validation.winner) - set(context['classes']):
            raise ValueError('Validation requires known labels in the declared classes.')
    return tuple(int(y) for y in sorted(years.unique())), validation_year


class custom_model(ABC):
    def __init__(self, name: str):
        self.name = name
        self.hyperparameters: dict = {}
        self.classes_ = np.asarray(PARTIES)
        self.training_records: list[dict] = []
        self.fit_record: dict | None = None

    @abstractmethod
    def train(self, data, configuration=None, fit_context=None) -> dict:
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
        self.classes_ = np.asarray(context['classes'])
        self.training_records = deepcopy(runs or [])
        self.fit_record = dict(training_years=years, configuration=deepcopy(configuration),
                               validation_year=validation_year, runs=deepcopy(self.training_records))
        return deepcopy(self.fit_record)

    def _align_probabilities(self, probabilities, fitted_classes):
        result = np.zeros((len(probabilities), len(self.classes_)), dtype=float)
        for index, party in enumerate(fitted_classes):
            matches = np.flatnonzero(self.classes_ == party)
            if not len(matches):
                raise ValueError(f'Undeclared fitted class: {party}')
            result[:, matches[0]] = probabilities[:, index]
        return result
