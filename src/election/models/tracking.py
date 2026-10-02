"""Explicit, synchronous MLflow summaries. No autologging or fitted artifacts."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import platform
import importlib.metadata
import inspect
import subprocess

import numpy as np
import pandas as pd
from mlflow import MlflowClient

CORE_METRICS = ('accuracy', 'changed_seat_accuracy', 'retained_seat_accuracy', 'macro_f1', 'log_loss')
COUNTS = ('evaluation_rows', 'changed_seat_evaluation_rows', 'retained_seat_evaluation_rows',
          'previous_winner_known_rows', 'test_rows', 'excluded_rows')


def clean(value):
    # Shared JSON conversion handles NumPy settings and explicit undefined scores.
    from election.models.evaluation import json_value
    return json_value(value)


class ExecutionTracking:
    def __init__(self, *, tracking_uri=None, experiment_name=None):
        uri = tracking_uri or os.environ.get('MLFLOW_TRACKING_URI')
        if not uri or not uri.startswith(('http://', 'https://')):
            raise ValueError('Set MLFLOW_TRACKING_URI to the PostgreSQL-backed tracking server; file/SQLite tracking is not supported.')
        self.client = MlflowClient(tracking_uri=uri)
        name = experiment_name or os.environ.get('MLFLOW_EXPERIMENT_NAME', 'election-prediction')
        experiment = self.client.get_experiment_by_name(name)
        self.experiment_id = experiment.experiment_id if experiment else self.client.create_experiment(name)
        self.parent_id = None
        self.active = False
        self.candidate_ids = {}
        self.running = set()

    def create(self, name, kind, parent=None, **tags):
        tags = {'mlflow.runName': str(name), 'kind': kind, **{k: str(v) for k, v in tags.items()}}
        if parent:
            tags['mlflow.parentRunId'] = parent
            tags['execution_id'] = self.parent_id
        run_id = self.client.create_run(self.experiment_id, tags=tags).info.run_id
        self.running.add(run_id)
        return run_id

    def artifact(self, run_id, value, filename):
        self.client.log_dict(run_id, clean(value), filename)

    def finish(self, run_id):
        self.client.set_tag(run_id, 'record_complete', 'true')
        self.client.set_terminated(run_id, 'FINISHED')
        self.running.remove(run_id)

    @contextmanager
    def execution(self, candidates, outer_elections, metadata):
        if self.active:
            raise RuntimeError('An execution is already active on this tracker')
        self.candidate_ids.clear()
        self.parent_id = self.create('pipeline execution', 'execution')
        self.active = True
        try:
            self.client.set_tag(self.parent_id, 'execution_id', self.parent_id)
            self.client.set_tag(self.parent_id, 'execution_status', 'running')
            self.artifact(self.parent_id, metadata, 'shared_configuration.json')
            self.artifact(self.parent_id, {'candidates': [s.name for s in candidates],
                          'outer_elections': outer_elections}, 'expected_evaluations.json')
            self.client.log_param(self.parent_id, 'selection_criterion', 'mean outer-election accuracy; equal election weights')
            self.client.log_param(self.parent_id, 'tie_rule', 'Exact ties follow registry order; inner ties follow configuration order')
            for order, spec in enumerate(candidates):
                run_id = self.create(spec.name, 'candidate', self.parent_id, candidate=spec.name, registry_order=order)
                self.candidate_ids[spec.name] = run_id
                factory = spec.factory
                from election import paths
                source_path = inspect.getsourcefile(factory)
                source_reference = str(Path(source_path).resolve().relative_to(paths.project_root())) if source_path and Path(source_path).resolve().is_relative_to(paths.project_root()) else factory.__module__
                specification = dict(name=spec.name, version=spec.version,
                    implementation=f'{factory.__module__}.{factory.__qualname__}',
                    implementation_reference=source_reference,
                    policy=spec.training_policy.version,
                    policy_reference='src/election/models/training_policy.py',
                    fixed_settings=spec.fixed_settings,
                    predictor_reference=metadata['predictor_guide'],
                    feature_preprocessing_reference=source_reference,
                    shared_configuration_run_id=self.parent_id)
                if 'hidden_sizes' in spec.fixed_settings:
                    specification['neural_structure'] = dict(hidden_sizes=spec.fixed_settings['hidden_sizes'],
                        hidden_activation='ReLU', output='linear logits; softmax probabilities',
                        ensemble='mean of per-seed probabilities', optimizer='SGD', loss='cross entropy',
                        input_dimension='determined by training-fitted preprocessing',
                        inner_checkpoint='lowest validation loss', refit='fresh networks; selected-fold per-seed median best epoch rounded half up')
                if spec.name == 'Conditional XGBoost':
                    specification['components'] = ['change probability', 'challenger role conditional on change']
                    specification['combination'] = 'P(hold) for incumbent; P(change) * P(role | change) for challengers'
                self.artifact(run_id, specification, 'candidate_specification.json')
            yield self
            final_runs = self.children(self.parent_id, 'final_test')
            if len(final_runs) != 1:
                raise RuntimeError('A successful execution requires one completed final test')
            self.validate(final_runs[0], CORE_METRICS)
            if any(self.client.get_run(run_id).info.status != 'FINISHED' for run_id in self.candidate_ids.values()):
                raise RuntimeError('A successful execution requires complete candidate summaries')
            self.client.set_tag(self.parent_id, 'execution_status', 'complete')
            self.finish(self.parent_id)
        except BaseException as error:
            failures = []
            for run_id in list(self.running):
                try:
                    self.client.set_tag(run_id, 'execution_status', 'failed')
                    self.client.set_tag(run_id, 'error', f'{type(error).__name__}: {error}'[:2000])
                    self.client.set_terminated(run_id, 'FAILED')
                    self.running.remove(run_id)
                except Exception as logging_error:
                    failures.append(str(logging_error))
            if failures:
                error.add_note('Unable to mark failed MLflow runs: ' + '; '.join(failures))
            raise
        finally:
            self.active = False

    @contextmanager
    def evaluation(self, name, *, candidate, election, final=False):
        run_id = self.create(name, 'final_test' if final else 'election',
                             self.parent_id if final else self.candidate_ids[candidate],
                             candidate=candidate, election=election,
                             candidate_specification_run_id=self.candidate_ids[candidate])
        yield run_id
        self.finish(run_id)

    def scores(self, run_id, values, *, summary=False):
        for metric in CORE_METRICS:
            key = f'mean_{metric}' if summary else metric
            value = values[key]
            if value is None:
                if metric not in ('changed_seat_accuracy', 'retained_seat_accuracy'):
                    raise ValueError(f'Undefined required metric {key}')
                self.client.set_tag(run_id, f'undefined.{key}', 'true')
            else:
                if not math.isfinite(value):
                    raise ValueError(f'Non-finite metric {key}')
                if (metric != 'log_loss' and not 0 <= value <= 1) or (metric == 'log_loss' and value < 0):
                    raise ValueError(f'Invalid metric {key}')
                self.client.log_metric(run_id, key, float(value))
        for key in COUNTS:
            if key in values:
                if int(values[key]) != values[key] or values[key] < 0:
                    raise ValueError(f'Invalid count {key}')
                self.client.log_param(run_id, key, int(values[key]))

    def children(self, parent, kind):
        # Iterate every page: large histories must never truncate the current run.
        result, token = [], None
        while True:
            page = self.client.search_runs([self.experiment_id],
                filter_string=f"tags.`mlflow.parentRunId` = '{parent}' and tags.kind = '{kind}'", page_token=token)
            result.extend(page)
            token = page.token
            if not token:
                return result

    def summarize_and_select(self, candidates, elections):
        summaries = self.children(self.parent_id, 'candidate')
        if len(summaries) != len(candidates) or {r.data.tags.get('candidate') for r in summaries} != {s.name for s in candidates}:
            raise RuntimeError('Missing or duplicate candidate summaries')
        for spec in candidates:
            run_id = self.candidate_ids[spec.name]
            evaluations = self.children(run_id, 'election')
            if len(evaluations) != len(elections) or sorted(int(r.data.tags['election']) for r in evaluations) != sorted(elections):
                raise RuntimeError('Incomplete outer-election comparison')
            for run in evaluations:
                self.validate(run, CORE_METRICS)
                if run.data.tags.get('execution_id') != self.parent_id or run.data.tags.get('candidate') != spec.name:
                    raise RuntimeError('Evaluation belongs to another execution/candidate')
            means, counts = {}, {}
            for metric in CORE_METRICS:
                values = [r.data.metrics[metric] for r in evaluations if metric in r.data.metrics]
                means[f'mean_{metric}'] = float(np.mean(values)) if values else None
                counts[f'{metric}_elections'] = len(values)
            self.scores(run_id, means, summary=True)
            for key, value in counts.items():
                self.client.log_param(run_id, key, value)
            self.finish(run_id)
        # Selection reads the persisted summaries again, never local calculated means.
        records = self.children(self.parent_id, 'candidate')
        if len(records) != len(candidates) or {r.data.tags.get('candidate') for r in records} != {s.name for s in candidates}:
            raise RuntimeError('Missing or duplicate completed candidate summaries')
        by_name = {r.data.tags['candidate']: r for r in records}
        for spec in candidates:
            run = by_name[spec.name]
            if run.data.tags.get('execution_id') != self.parent_id or run.info.run_id != self.candidate_ids[spec.name]:
                raise RuntimeError('Candidate summary belongs to another execution')
            if int(run.data.params.get('accuracy_elections', -1)) != len(elections):
                raise RuntimeError('Incomplete candidate accuracy summary')
            self.validate(run, [f'mean_{m}' for m in CORE_METRICS])
        winner = max(candidates, key=lambda s: by_name[s.name].data.metrics['mean_accuracy'])
        tied = [s.name for s in candidates if by_name[s.name].data.metrics['mean_accuracy'] == by_name[winner.name].data.metrics['mean_accuracy']]
        self.client.set_tag(self.parent_id, 'winning_candidate', winner.name)
        self.client.set_tag(self.parent_id, 'winning_candidate_run_id', self.candidate_ids[winner.name])
        self.client.set_tag(self.parent_id, 'tie_decision', json.dumps({'tied': tied, 'selected': winner.name}))
        return winner

    @staticmethod
    def validate(run, keys):
        if run.info.status != 'FINISHED' or run.data.tags.get('record_complete') != 'true':
            raise RuntimeError('Incomplete MLflow record')
        for key in keys:
            value = run.data.metrics.get(key)
            if value is None and key.endswith(('changed_seat_accuracy', 'retained_seat_accuracy')) and run.data.tags.get(f'undefined.{key}') == 'true':
                continue
            if value is None or not math.isfinite(value) or (not key.endswith('log_loss') and not 0 <= value <= 1) or (key.endswith('log_loss') and value < 0):
                raise RuntimeError(f'Invalid selection record: {key}')

    def scorecard(self):
        rows = [dict(candidate=r.data.tags['candidate'],
                     **{f'mean_{m}': r.data.metrics.get(f'mean_{m}') for m in CORE_METRICS},
                     **{f'{m}_elections': int(r.data.params[f'{m}_elections']) for m in CORE_METRICS})
                for r in self.children(self.parent_id, 'candidate') if r.info.status == 'FINISHED']
        order = {name: i for i, name in enumerate(self.candidate_ids)}
        rows.sort(key=lambda r: (-r['mean_accuracy'], order[r['candidate']]))
        return pd.DataFrame(rows)

    def election_scores(self):
        return [dict(candidate=name, election=int(r.data.tags['election']),
                     **{m: r.data.metrics.get(m) for m in CORE_METRICS},
                     **{k: int(v) for k, v in r.data.params.items() if k in COUNTS})
                for name, parent in self.candidate_ids.items() for r in self.children(parent, 'election') if r.info.status == 'FINISHED']


def execution_metadata(*, inner_elections, outer_elections, classes, forecast_election, predictor_guide):
    from election import paths
    root = paths.project_root()
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git('status', '--porcelain')
    versions = {}
    for package in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'torch', 'mlflow', 'psycopg2-binary', 'duckdb'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(git_commit=git('rev-parse', 'HEAD'), git_dirty=None if status is None else bool(status),
        python=platform.python_version(), packages=versions, dependency_reference='pyproject.toml',
        dataset_reference='src/election/data/README.md', predictor_guide=str(Path(predictor_guide).resolve()),
        predictor_generation_reference='src/election/pipeline/helpers/predictor_guide.py',
        classes=classes, inner_elections=inner_elections, outer_elections=outer_elections,
        forecast_election=forecast_election,
        split_rule='whole-election validation; training strictly earlier; final training all pre-forecast labelled history',
        splitting_seed=None, tuning_seed=None, tuning='deterministic ParameterGrid in registry order',
        aggregation='equal election means; omit undefined subgroup scores; exclude unknown outcomes; macro F1 uses all declared classes')
