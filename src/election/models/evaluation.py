"""Nested whole-election evaluation with temporary tuning and MLflow summaries."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import election.models.custom_model as custom_model_module

OUTER_ELECTIONS = (2005, 2010, 2015, 2017, 2019)
INNER_ELECTIONS = (1997, 2001, 2005, 2010, 2015, 2017, 2019)


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


def _cache_signature(frame):
    """Existing content hashes are temporary cache keys, never dataset records."""
    digest = sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    digest.update(str(list(frame.columns)).encode())
    digest.update(str(list(frame.dtypes.astype(str))).encode())
    return digest.hexdigest()


def metrics(frame, probabilities, classes):
    actual = frame.winner.to_numpy()
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
    known_previous = frame.previous_winner.notna().to_numpy()
    previous_correct = frame.previous_winner.eq(frame.winner).fillna(False).to_numpy(dtype=bool)
    changed = known_previous & ~previous_correct
    retained = known_previous & previous_correct
    indices = pd.Index(classes).get_indexer(actual)
    if (indices < 0).any():
        raise ValueError('Observed party is outside the declared class convention.')
    correct = predicted == actual
    result = {
        'accuracy': float(correct.mean()), 'evaluation_rows': len(frame),
        'changed_seat_accuracy': float(correct[changed].mean()) if changed.any() else None,
        'changed_seat_evaluation_rows': int(changed.sum()),
        'retained_seat_accuracy': float(correct[retained].mean()) if retained.any() else None,
        'retained_seat_evaluation_rows': int(retained.sum()),
        'macro_f1': float(f1_score(actual, predicted, labels=list(classes),
                                   average='macro', zero_division=0)),
        'log_loss': float(-np.log(np.clip(probabilities[np.arange(len(frame)), indices],
                                         np.finfo(float).eps, 1)).mean()),
        # Missing previous winners are incorrect baseline forecasts, not dropped rows.
        'previous_winner_accuracy': float(previous_correct.mean()),
        'previous_winner_known_rows': int(known_previous.sum()),
    }
    for party in classes:
        result[f'seat_error_{party}'] = int((predicted == party).sum() - (actual == party).sum())
    return result


@dataclass
class EvaluationReport:
    tracking: object
    status: str = 'incomplete'
    error: str | None = None
    selected_candidate: str | None = None
    final_fit: dict | None = None
    final_metrics: dict | None = None

    @property
    def execution_id(self):
        return self.tracking.parent_id

    def scorecard(self):
        return self.tracking.scorecard()

    @property
    def outer_results(self):
        return self.tracking.election_scores()


@dataclass
class SelectionResult:
    model: object
    report: EvaluationReport

    @property
    def model_name(self):
        return self.model.name

    def __iter__(self):
        # Preserve `model, model_name = run_pipeline(...)` for existing callers.
        yield self.model
        yield self.model_name

    def __getitem__(self, index):
        return (self.model, self.model_name)[index]


class HistoricalEvaluator:
    """No model-type branches: each CandidateSpec supplies its fitting policy."""
    def __init__(self, candidates, *, outer_elections=OUTER_ELECTIONS,
                 inner_elections=INNER_ELECTIONS, classes=custom_model_module.PARTIES,
                 verbose=True):
        self.candidates = tuple(candidates)
        self.outer_elections = tuple(outer_elections)
        self.inner_elections = tuple(inner_elections)
        self.classes = tuple(classes)
        self.verbose = verbose
        if not self.candidates or len({s.name for s in self.candidates}) != len(self.candidates):
            raise ValueError('Provide nonempty, uniquely named candidates.')
        for years in (self.outer_elections, self.inner_elections):
            if not years or tuple(sorted(set(years))) != years:
                raise ValueError('Election schedules must be nonempty, unique and ascending.')
        if len(set(self.classes)) != len(self.classes) or not self.classes:
            raise ValueError('Party classes must be nonempty and unique.')
        self.report = None
        self._inner_cache = {}

    def _prepare(self, data):
        required = {'election', 'winner', 'previous_winner'}
        for spec in self.candidates:
            required.update(spec.feature_columns)
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f'Missing required data columns: {sorted(missing)}')
        frame = data.copy(deep=True)
        years = pd.to_numeric(frame.election, errors='raise')
        if not np.isfinite(years).all() or not (years == years.astype(int)).all():
            raise ValueError('Election years must be finite integers.')
        frame['election'] = years.astype(int)
        frame = frame.dropna(subset=['winner'])
        if frame.empty or set(frame.winner) - set(self.classes):
            raise ValueError('Expected labelled rows with declared party classes.')
        return frame

    def _inner_years(self, history, forecast_election=None):
        # All scheduled inner elections before the forecast date must be present.
        if history.empty:
            raise ValueError('No labelled history is available for inner tuning.')
        cutoff = forecast_election if forecast_election is not None else history.election.max() + 1
        expected = [v for v in self.inner_elections if v < cutoff]
        available = set(history.election)
        missing = set(expected) - available
        if missing or len(expected) < 2:
            raise ValueError(f'At least two scheduled inner elections are required; missing {sorted(missing)}.')
        for v in expected:
            if history.loc[history.election < v, 'election'].nunique() < 2:
                raise ValueError(f'Inner election {v} needs at least two earlier training elections.')
        return expected

    @staticmethod
    def _model_frame(spec, data, *, labelled=False):
        columns = [*spec.feature_columns, 'election']
        if labelled:
            columns.append('winner')
        return data[list(dict.fromkeys(columns))].copy()

    def _probabilities(self, model, data, spec):
        # Only declared predictors and election metadata reach prediction methods.
        predictors = self._model_frame(spec, data)
        values = np.asarray(model.predict_proba(predictors), dtype=float)
        if tuple(model.classes_) != self.classes:
            raise ValueError(f'{model.name}: probability columns do not match declared party order.')
        if values.shape != (len(data), len(self.classes)) or not np.isfinite(values).all():
            raise ValueError(f'{model.name}: invalid probability shape or non-finite probabilities.')
        if (values < 0).any() or not np.allclose(values.sum(axis=1), 1, atol=1e-6):
            raise ValueError(f'{model.name}: probabilities must be nonnegative and sum to one.')
        return values

    def _tune(self, spec, history, forecast_election):
        years = self._inner_years(history, forecast_election)
        configurations = spec.configurations()
        if not configurations:
            raise ValueError(f'{spec.name}: empty search space.')
        folds = {}
        for v in years:
            training = history.loc[history.election < v].copy()
            validation = history.loc[history.election == v].copy()
            folds[v] = (training, validation, _cache_signature(training),
                        _cache_signature(validation), {})
        best_score, best_configuration, best_records = -np.inf, None, None
        last_progress = perf_counter()
        for ci, configuration in enumerate(configurations):
            scores, records = [], []
            config_key = json.dumps(json_value(configuration), sort_keys=True)
            for v, (training, validation, train_hash, valid_hash, stage_cache) in folds.items():
                # In-memory only. The dataset, code and registry are fixed for this run.
                key = (id(spec), spec.version, spec.training_policy.version, self.classes,
                       config_key, train_hash, valid_hash)
                if key not in self._inner_cache:
                    model = spec.create_model()
                    context = spec.training_policy.inner_context(
                        self._model_frame(spec, validation, labelled=True), self.classes, stage_cache)
                    record = model.train(self._model_frame(spec, training, labelled=True), deepcopy(configuration), context)
                    probabilities = self._probabilities(model, validation, spec)
                    score = float((np.asarray(self.classes)[probabilities.argmax(axis=1)] == validation.winner).mean())
                    self._inner_cache[key] = (score, deepcopy(record))
                score, record = self._inner_cache[key]
                scores.append(score)
                records.append(deepcopy(record))
            average = float(np.mean(scores))
            # Exact ties follow the predefined configuration order.
            if average > best_score:
                best_score, best_configuration, best_records = average, deepcopy(configuration), records
            if self.verbose and perf_counter()-last_progress >= 30:
                print(f'  {spec.name}: {ci+1}/{len(configurations)} configurations scored', flush=True)
                last_progress = perf_counter()
        return best_configuration, best_records, best_score

    def tune_and_refit(self, spec, history, *, forecast_election=None):
        history = self._prepare(history)
        if forecast_election is not None and (history.election >= forecast_election).any():
            raise ValueError('Forecast history must contain only earlier elections.')
        config, records, _ = self._tune(spec, history, forecast_election)
        context = spec.training_policy.refit_context(records, self.classes)
        model = spec.create_model()
        record = model.train(self._model_frame(spec, history, labelled=True), config, context)
        self._last_refit = dict(candidate=spec.name, forecast_election=forecast_election,
                               configuration=deepcopy(model.hyperparameters),
                               training_years=record.training_years,
                               refit_durations=deepcopy(context.durations))
        return model

    def evaluate(self, data, *, forecast_election=2024, test_data=None,
                 output_dir=None, tracking=None, predictor_guide=None):
        from election.models.tracking import ExecutionTracking, execution_metadata
        from election import paths
        tracking = tracking or ExecutionTracking()
        output_dir = Path(output_dir) if output_dir is not None else paths.project_root() / 'notebooks' / 'outputs'
        self.report = EvaluationReport(tracking)
        self._inner_cache.clear()
        metadata = execution_metadata(inner_elections=self.inner_elections,
            outer_elections=self.outer_elections, classes=self.classes,
            forecast_election=forecast_election,
            predictor_guide=predictor_guide or output_dir / 'predictor_descriptions.md')
        try:
            scope = nullcontext(tracking) if tracking.active else tracking.execution(self.candidates, self.outer_elections, metadata)
            with scope:
                if test_data is None:
                    raise ValueError('Final test data is required for a complete execution.')
                if (pd.to_numeric(data['election'], errors='raise') >= forecast_election).any():
                    raise ValueError('Input includes the forecast election or later rows.')
                frame = self._prepare(data)
                if any(e not in set(frame.election) for e in self.outer_elections):
                    raise ValueError('Every outer evaluation election must have labelled rows.')
                for e in self.outer_elections:
                    self._inner_years(frame.loc[frame.election < e], e)
                self._inner_years(frame, forecast_election)
                for e in self.outer_elections:
                    history = frame.loc[frame.election < e]
                    validation = frame.loc[frame.election == e]
                    for spec in self.candidates:
                        if self.verbose:
                            print(f'Outer election {e}: tuning and fitting {spec.name}', flush=True)
                        with tracking.evaluation(f'{spec.name}: {e}', candidate=spec.name, election=e) as run_id:
                            model = self.tune_and_refit(spec, history, forecast_election=e)
                            tracking.artifact(run_id, self._last_refit, 'selected_configuration.json')
                            probabilities = self._probabilities(model, validation, spec)
                            tracking.scores(run_id, metrics(validation, probabilities, self.classes))
                        if self.verbose:
                            print(f'Completed {spec.name}: {e}', flush=True)
                spec = tracking.summarize_and_select(self.candidates, self.outer_elections)
                self.report.selected_candidate = spec.name
                if self.verbose:
                    print(self.report.scorecard().to_string(index=False), flush=True)
                    print(f'Selected {spec.name}; final tuning, fitting and {forecast_election} evaluation', flush=True)
                with tracking.evaluation(f'Final test: {forecast_election}', candidate=spec.name,
                                         election=forecast_election, final=True) as run_id:
                    model = self.tune_and_refit(spec, frame, forecast_election=forecast_election)
                    self.report.final_fit = deepcopy(self._last_refit)
                    tracking.artifact(run_id, self._last_refit, 'selected_configuration.json')
                    final_metrics = self._final_evaluation(model, spec, test_data, forecast_election, output_dir)
                    tracking.scores(run_id, final_metrics)
                    self.report.final_metrics = final_metrics
            self.report.status = 'complete'
            return SelectionResult(model, self.report)
        except BaseException as error:
            self.report.status = 'failed'
            self.report.error = f'{type(error).__name__}: {error}'
            raise
        finally:
            self._inner_cache.clear()

    def _final_evaluation(self, model, spec, test_data, forecast_election, output_dir):
        from sklearn.metrics import confusion_matrix
        from matplotlib.figure import Figure
        from sklearn.metrics import ConfusionMatrixDisplay
        if test_data.empty or not (pd.to_numeric(test_data.election, errors='raise') == forecast_election).all():
            raise ValueError('Final test must contain only the forecast election.')
        labelled = self._prepare(test_data)
        # Predict all seats; only unknown outcomes are excluded from scores.
        all_probabilities = self._probabilities(model, test_data, spec)
        predicted = np.asarray(self.classes)[all_probabilities.argmax(axis=1)]
        probabilities = self._probabilities(model, labelled, spec)
        values = metrics(labelled, probabilities, self.classes)
        values.update(test_rows=len(test_data), excluded_rows=len(test_data)-len(labelled))
        output_dir.mkdir(parents=True, exist_ok=True)
        export = test_data.copy()
        export['predicted_winner'] = predicted
        export.to_csv(output_dir / 'test_predictions.csv', index=False)
        labelled_predictions = np.asarray(self.classes)[probabilities.argmax(axis=1)]
        labels = sorted(set(labelled.winner) | set(labelled_predictions))
        matrix = confusion_matrix(labelled.winner, labelled_predictions, labels=labels)
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
            output_dir / 'test_confusion_matrix.csv', index_label='actual_winner')
        fig = Figure(figsize=(8, 6))
        ax = fig.subplots()
        ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
        ax.set_title(f'{model.name}: {forecast_election} retrospective confusion matrix')
        ax.set_xlabel('Predicted winning party')
        ax.set_ylabel('Actual winning party')
        fig.tight_layout()
        fig.savefig(output_dir / 'test_confusion_matrix.png', dpi=150, bbox_inches='tight')
        return values
