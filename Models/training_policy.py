"""Model-specific refit rules; the evaluator has no neural-model type checks."""
from dataclasses import dataclass
import math
import statistics

from Models.custom_model import FitContext


@dataclass(frozen=True)
class TrainingPolicy:
    version: str = 'full-history-v1'

    def inner_context(self, validation, classes, cache):
        # Ordinary estimators do not need validation outcomes to fit.
        return FitContext(classes=classes, cache=cache)

    def refit_context(self, records, classes):
        return FitContext(classes=classes)


@dataclass(frozen=True)
class NeuralTrainingPolicy(TrainingPolicy):
    version: str = 'per-seed-median-best-epoch-v1'

    def inner_context(self, validation, classes, cache):
        return FitContext(validation=validation, classes=classes, cache=cache)

    def refit_context(self, records, classes):
        if not records:
            raise ValueError('Neural refitting requires inner checkpoint records.')
        seed_sets = [{run['seed'] for run in r.runs} for r in records]
        if not seed_sets[0] or any(seeds != seed_sets[0] for seeds in seed_sets):
            raise ValueError('Every inner election must have the same neural seeds.')
        durations = {}
        for seed in sorted(seed_sets[0]):
            epochs = []
            for record in records:
                runs = [run for run in record.runs if run['seed'] == seed]
                if len(runs) != 1 or not runs[0]['best_epoch'] or runs[0]['best_epoch'] < 1:
                    raise ValueError('Expected one positive best epoch per fold and seed.')
                epochs.append(runs[0]['best_epoch'])
            durations[seed] = max(1, math.floor(statistics.median(epochs) + 0.5))
        return FitContext(classes=classes, durations=durations)
