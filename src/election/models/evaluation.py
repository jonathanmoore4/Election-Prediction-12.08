"""Nested, whole-election evaluation with reusable inner predictions and audit reports."""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import platform
import importlib.metadata

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


def fingerprint(frame):
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
    outer_elections: tuple[int, ...]
    metadata: dict = field(default_factory=dict)
    inner_results: list[dict] = field(default_factory=list)
    outer_results: list[dict] = field(default_factory=list)
    predictions: list[dict] = field(default_factory=list)
    final_fit: dict | None = None
    selected_candidate: str | None = None
    status: str = 'incomplete'
    error: str | None = None

    def scorecard(self):
        rows = []
        order = self.metadata.get('candidate_order', [])
        for name in order:
            results = [r for r in self.outer_results if r['candidate'] == name]
            if sorted(r['election'] for r in results) != sorted(self.outer_elections):
                continue  # Never present an average with a missing election as complete.
            accuracies = [r['accuracy'] for r in results]
            row = dict(candidate=name, mean_accuracy=float(np.mean(accuracies)),
                       worst_accuracy=min(accuracies), best_accuracy=max(accuracies),
                       accuracy_range=max(accuracies)-min(accuracies),
                       latest_three_accuracy=float(np.mean([r['accuracy'] for r in
                                                            sorted(results, key=lambda r: r['election'])[-3:]])),
                       runtime_seconds=sum(r['runtime_seconds'] for r in results))
            for metric in ['changed_seat_accuracy', 'retained_seat_accuracy', 'macro_f1',
                           'log_loss', 'previous_winner_accuracy', 'seed_accuracy_std']:
                values = [r[metric] for r in results if r[metric] is not None]
                row[f'mean_{metric}'] = float(np.mean(values)) if values else None
                row[f'{metric}_elections'] = len(values)
            for r in results:
                row[f'accuracy_{r["election"]}'] = r['accuracy']
            rows.append(row)
        if not rows:
            return pd.DataFrame(columns=['candidate', 'mean_accuracy'])
        card = pd.DataFrame(rows).sort_values('mean_accuracy', ascending=False, kind='stable').reset_index(drop=True)
        card['comparison_complete'] = len(rows) == len(order)
        card['report_status'] = self.status
        return card

    def save(self, scores_path):
        """Write a summary, per-election scores, predictions and full fit audit."""
        path = Path(scores_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stem = path.with_suffix('')
        self.scorecard().to_csv(path, index=False)
        pd.DataFrame(self.outer_results).to_csv(str(stem) + '_elections.csv', index=False)
        pd.DataFrame(self.predictions).to_csv(str(stem) + '_predictions.csv', index=False)
        target = Path(str(stem) + '_report.json')
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(json_value(asdict(self)), indent=2, allow_nan=False) + '\n')
        temporary.replace(target)


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
                 inner_elections=INNER_ELECTIONS, classes=custom_model_module.PARTIES, scores_path=None,
                 verbose=True):
        self.candidates = tuple(candidates)
        self.outer_elections = tuple(outer_elections)
        self.inner_elections = tuple(inner_elections)
        self.classes = tuple(classes)
        self.scores_path = scores_path
        self.verbose = verbose
        if not self.candidates or len({s.name for s in self.candidates}) != len(self.candidates):
            raise ValueError('Provide nonempty, uniquely named candidates.')
        for years in (self.outer_elections, self.inner_elections):
            if not years or tuple(sorted(set(years))) != years:
                raise ValueError('Election schedules must be nonempty, unique and ascending.')
        if len(set(self.classes)) != len(self.classes) or not self.classes:
            raise ValueError('Party classes must be nonempty and unique.')
        self.report = EvaluationReport(self.outer_elections)
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
        frame['_evaluation_row_id'] = np.arange(len(frame))
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
            folds[v] = (training, validation, fingerprint(training.drop(columns=['_evaluation_row_id'])),
                        fingerprint(validation.drop(columns=['_evaluation_row_id'])), {})
        best_score, best_configuration, best_records = -np.inf, None, None
        last_progress = perf_counter()
        for ci, configuration in enumerate(configurations):
            scores, records = [], []
            config_key = json.dumps(json_value(configuration), sort_keys=True)
            for v, (training, validation, train_hash, valid_hash, stage_cache) in folds.items():
                # In-memory only. The dataset, code and registry are fixed for this run.
                key = (id(spec), spec.version, spec.training_policy.version, self.classes,
                       config_key, train_hash, valid_hash)
                self.report.metadata['active_fit'] = dict(candidate=spec.name, forecast_election=forecast_election,
                                                          validation_election=v, configuration=configuration)
                reused = key in self._inner_cache
                if not reused:
                    model = spec.create_model()
                    context = spec.training_policy.inner_context(
                        self._model_frame(spec, validation, labelled=True), self.classes, stage_cache)
                    start = perf_counter()
                    record = model.train(self._model_frame(spec, training, labelled=True), deepcopy(configuration), context)
                    probabilities = self._probabilities(model, validation, spec)
                    score = float((np.asarray(self.classes)[probabilities.argmax(axis=1)] == validation.winner).mean())
                    self._inner_cache[key] = (score, deepcopy(record), perf_counter()-start)
                score, record, runtime = self._inner_cache[key]
                scores.append(score)
                records.append(deepcopy(record))
                self.report.inner_results.append(dict(candidate=spec.name,
                    forecast_election=forecast_election, validation_election=v,
                    configuration=deepcopy(configuration), accuracy=score,
                    fit_record=asdict(record), reused=reused, fit_seconds=runtime))
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
        config, records, score = self._tune(spec, history, forecast_election)
        context = spec.training_policy.refit_context(records, self.classes)
        self.report.metadata['active_fit'] = dict(candidate=spec.name, forecast_election=forecast_election,
                                                  phase='refit', configuration=config)
        model = spec.create_model()
        record = model.train(self._model_frame(spec, history, labelled=True), config, context)
        self._last_refit = dict(candidate=spec.name, forecast_election=forecast_election,
                               configuration=deepcopy(config), inner_accuracy=score,
                               fit_record=asdict(record))
        return model

    def evaluate(self, data, *, forecast_election=2024):
        # Each evaluation starts a new report and cache; inner reuse never crosses runs.
        self.report = EvaluationReport(self.outer_elections)
        self._inner_cache.clear()
        try:
            if (pd.to_numeric(data['election'], errors='raise') >= forecast_election).any():
                raise ValueError('Input includes the forecast election or later rows.')
            frame = self._prepare(data)
            if (frame.election >= forecast_election).any():
                raise ValueError('Input includes the forecast election or later outcomes.')
            if any(e not in set(frame.election) for e in self.outer_elections):
                raise ValueError('Every outer evaluation election must have labelled rows.')
            # Fail before expensive training if any fold is impossible.
            for e in self.outer_elections:
                self._inner_years(frame.loc[frame.election < e], e)
            self._inner_years(frame, forecast_election)
            source = sha256()
            model_root = Path(__file__).parent
            for path in sorted(model_root.rglob('*.py')):
                source.update(path.relative_to(model_root).as_posix().encode())
                source.update(path.read_bytes())
            versions = {}
            for package in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'torch'):
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    pass
            self.report.metadata = dict(data_version=fingerprint(frame), code_version=source.hexdigest(),
                python=platform.python_version(), packages=versions, classes=self.classes,
                inner_elections=self.inner_elections, forecast_election=forecast_election,
                candidate_order=[s.name for s in self.candidates],
                tie_rule='Exact ties use registry/configuration order; no practical-tie threshold.',
                candidates=[dict(name=s.name, version=s.version, policy=s.training_policy.version,
                                 search_space=s.search_space, fixed_settings=s.fixed_settings,
                                 feature_columns=s.feature_columns) for s in self.candidates])
            for e in self.outer_elections:
                history = frame.loc[frame.election < e]
                validation = frame.loc[frame.election == e]
                for spec in self.candidates:
                    if self.verbose:
                        print(f'Outer election {e}: tuning and fitting {spec.name}', flush=True)
                    start = perf_counter()
                    model = self.tune_and_refit(spec, history, forecast_election=e)
                    probabilities = self._probabilities(model, validation, spec)
                    result = dict(candidate=spec.name, election=e,
                                  **metrics(validation, probabilities, self.classes),
                                  runtime_seconds=perf_counter()-start,
                                  **{k: v for k, v in self._last_refit.items()
                                     if k not in ('candidate', 'forecast_election')})
                    members = model.member_probabilities(self._model_frame(spec, validation))
                    result['seed_metrics'] = {
                        seed: metrics(validation, values, self.classes)
                        for seed, values in members.items()}
                    seed_scores = [m['accuracy'] for m in result['seed_metrics'].values()]
                    result['seed_accuracy_std'] = float(np.std(seed_scores)) if seed_scores else None
                    self.report.outer_results.append(result)
                    predicted = np.asarray(self.classes)[probabilities.argmax(axis=1)]
                    for position, (_, row) in enumerate(validation.iterrows()):
                        saved = dict(candidate=spec.name, election=e,
                                     row_id=int(row['_evaluation_row_id']), actual_winner=row.winner,
                                     predicted_winner=predicted[position],
                                     configuration=json.dumps(json_value(model.hyperparameters), sort_keys=True))
                        for column in ('constituency_id', 'constituency_name', 'boundary_set'):
                            if column in row:
                                saved[column] = row[column]
                        saved.update({f'probability_{party}': float(probabilities[position, i])
                                      for i, party in enumerate(self.classes)})
                        self.report.predictions.append(saved)
                    if self.scores_path:
                        self.report.save(self.scores_path)
            scorecard = self.report.scorecard()
            if len(scorecard) != len(self.candidates):
                raise RuntimeError('Incomplete comparison: every candidate needs every outer election.')
            name = scorecard.iloc[0].candidate
            spec = next(s for s in self.candidates if s.name == name)
            self.report.selected_candidate = name
            # Leave-one-election-out ranking is a diagnostic, never a selection override.
            self.report.metadata['leave_one_out_winners'] = {
                e: max(self.candidates, key=lambda s: np.mean([
                    r['accuracy'] for r in self.report.outer_results
                    if r['candidate'] == s.name and r['election'] != e])).name
                for e in self.outer_elections if len(self.outer_elections) > 1}
            if self.verbose:
                print(scorecard.to_string(index=False), flush=True)
                print(f'Selected {name}; tuning and refitting before {forecast_election}', flush=True)
            model = self.tune_and_refit(spec, frame, forecast_election=forecast_election)
            self.report.final_fit = deepcopy(self._last_refit)
            self.report.metadata.pop('active_fit', None)
            self.report.status = 'complete'
            return SelectionResult(model, self.report)
        except Exception as error:
            self.report.status = 'failed'
            self.report.error = f'{type(error).__name__}: {error}'
            raise
        finally:
            if self.scores_path:
                self.report.save(self.scores_path)
