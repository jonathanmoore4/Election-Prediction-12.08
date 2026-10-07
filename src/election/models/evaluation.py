"""Functions for nested whole-election evaluation; no concrete model imports."""
from copy import deepcopy
from hashlib import sha256
import inspect
import json
from time import perf_counter

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import ParameterGrid

from election.models.config import (
    OUTER_ELECTIONS, INNER_ELECTIONS, PARTIES, MODEL_COMPARISON_SCHEDULE,
    SCORING_ID, TARGET_ID, EVALUATION_PROTOCOL,
)


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if value is pd.NA:
        return None
    return value


def data_identity(data):
    """Content identity, not a dataset upload or a promise of data preservation."""
    digest = sha256(pd.util.hash_pandas_object(data, index=False).values.tobytes())
    digest.update(str(list(data.columns)).encode())
    digest.update(str(list(data.dtypes.astype(str))).encode())
    return digest.hexdigest()


def prepared_frame(data):
    missing = {'election', 'winner'} - set(data.columns)
    if missing:
        raise ValueError(f'Missing required data columns: {sorted(missing)}')
    frame = data.copy()
    years = pd.to_numeric(frame.election, errors='raise')
    if not np.isfinite(years).all() or not (years == years.astype(int)).all():
        raise ValueError('Election years must be finite integers.')
    frame['election'] = years.astype(int)
    frame = frame.dropna(subset=['winner'])
    if frame.empty or set(frame.winner) - set(PARTIES):
        raise ValueError('Expected labelled rows with declared party classes.')
    return frame


def split_before_election(data, election, *, training_cutoff=None):
    years = pd.to_numeric(data.election, errors='raise')
    eligible = years < election
    if training_cutoff is not None:
        eligible &= years <= training_cutoff
    train = data.loc[eligible & data.winner.notna()].copy()
    test = data.loc[(years == election) & data.winner.notna()].copy()
    if train.empty or test.empty:
        raise ValueError(f'Election {election}: training and test partitions must be nonempty.')
    return train, test


def _inner_elections(schedule, outer):
    mapping = schedule['inner_elections']
    return mapping.get(outer, mapping.get(str(outer), []))


def validate_schedule(data, schedule):
    frame = prepared_frame(data)
    outer = schedule['outer_elections']
    if not outer or list(sorted(set(outer))) != list(outer):
        raise ValueError('Outer elections must be nonempty, unique and ascending.')
    cutoff = schedule.get('training_cutoff')
    if cutoff is not None and (not isinstance(cutoff, int) or isinstance(cutoff, bool)):
        raise ValueError('Training cutoff must be an integer election year or None.')
    if schedule.get('purpose') not in ('model_comparison', 'final_evaluation'):
        raise ValueError('Schedule purpose must be model_comparison or final_evaluation.')
    if schedule['purpose'] == 'final_evaluation' and (list(outer) != [2024] or cutoff != 2019):
        raise ValueError('Final evaluation reserves 2024 and requires a 2019 training cutoff.')
    for election in outer:
        history, _ = split_before_election(frame, election, training_cutoff=cutoff)
        inner = _inner_elections(schedule, election)
        if len(inner) < 2 or list(sorted(set(inner))) != list(inner):
            raise ValueError(f'Election {election}: at least two ascending inner elections required.')
        if any(v >= election or (cutoff is not None and v > cutoff) for v in inner):
            raise ValueError('Inner elections must precede outer elections and respect the cutoff.')
        for year in inner:
            training, _ = split_before_election(history, year, training_cutoff=cutoff)
            if training.election.nunique() < 2:
                raise ValueError(f'Inner election {year} needs at least two earlier training elections.')


def parameter_combinations(hyperparameter_candidates):
    """Use existing ParameterGrid ordering for deterministic first-maximum ties."""
    return [deepcopy(values) for values in ParameterGrid(hyperparameter_candidates)]


def _evaluate(model_function, training, test, parameters, **options):
    # Plain three-argument user functions also work. Built-ins accept transient
    # checkpoint records for refits and can return a richer internal dictionary.
    signature = inspect.signature(model_function)
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())
    kwargs = {k: v for k, v in options.items()
              if accepts_kwargs or k in signature.parameters}
    value = model_function(training, test, deepcopy(parameters), **kwargs)
    result = value if isinstance(value, dict) else {'accuracy': value}
    score = result.get('accuracy')
    if isinstance(score, bool) or not isinstance(score, (float, int, np.number)) or not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Model evaluation must return a finite accuracy between zero and one.')
    return {**result, 'accuracy': float(score)}


def suggest_parameters(trial, hyperparameter_candidates):
    """Sample any model's existing discrete search space through one interface."""
    return {name: trial.suggest_categorical(name, list(values))
            for name, values in sorted(hyperparameter_candidates.items())}


def tune_hyperparameters(model_function, hyperparameter_candidates, data, inner_elections,
                         *, training_cutoff=None, fixed_settings=None, verbose=True,
                         _fit_cache=None, n_trials=30, random_seed=42):
    """Maximise mean inner-election accuracy with seeded Optuna random search."""
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError('n_trials must be a positive integer.')
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError('random_seed must be a nonnegative integer.')
    for name, values in hyperparameter_candidates.items():
        if not isinstance(values, (list, tuple, np.ndarray)) or len(values) == 0:
            raise ValueError(f'{name}: candidates must be a nonempty sequence.')
    frame = prepared_frame(data)
    years = list(inner_elections)
    if len(years) < 2 or sorted(set(years)) != years:
        raise ValueError('At least two ascending inner elections are required.')
    if training_cutoff is not None and any(year > training_cutoff for year in years):
        raise ValueError('Inner elections exceed the training cutoff.')
    folds = {}
    for year in years:
        training, validation = split_before_election(frame, year, training_cutoff=training_cutoff)
        if training.election.nunique() < 2:
            raise ValueError(f'Inner election {year} needs at least two earlier training elections.')
        folds[year] = (training, validation, data_identity(training), data_identity(validation), {})
    fit_cache = {} if _fit_cache is None else _fit_cache
    last_progress = perf_counter()

    def objective(trial):
        nonlocal last_progress
        candidate = suggest_parameters(trial, hyperparameter_candidates)
        configuration = {**deepcopy(fixed_settings or {}), **candidate}
        scores, records = {}, []
        config_key = json.dumps(json_value(configuration), sort_keys=True, allow_nan=False)
        for year, (training, validation, train_id, valid_id, stage_cache) in folds.items():
            key = (model_function, config_key, train_id, valid_id)
            if key not in fit_cache:
                details = _evaluate(model_function, training, validation, configuration,
                                    cache=stage_cache, return_details=True)
                # Only checkpoint records/scores are retained; not fitted ensembles.
                fit_cache[key] = (details['accuracy'], deepcopy(details.get('fit_record')))
            score, record = fit_cache[key]
            scores[year] = score
            if record is not None:
                records.append(deepcopy(record))
        mean = float(np.mean(list(scores.values())))
        trial.set_user_attr('evaluation', dict(best_hyperparameters=deepcopy(candidate),
            mean_inner_accuracy=mean, inner_accuracies=scores, fit_records=records))
        if verbose and perf_counter() - last_progress >= 30:
            print(f'  {model_function.__name__}: {trial.number + 1}/{n_trials} trials scored', flush=True)
            last_progress = perf_counter()
        return mean

    # A fixed configuration needs only one trial. Duplicate sampled configurations
    # reuse scores and checkpoint records through the existing fit cache.
    trial_count = 1 if all(len(v) == 1 for v in hyperparameter_candidates.values()) else n_trials
    verbosity = optuna.logging.get_verbosity()
    try:
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize',
            sampler=optuna.samplers.RandomSampler(seed=random_seed))
        study.optimize(objective, n_trials=trial_count)
    finally:
        optuna.logging.set_verbosity(verbosity)
    return deepcopy(study.best_trial.user_attrs['evaluation'])


def nested_cv(model_function, hyperparameter_candidates, data, *, model_id, metadata=None,
              schedule=MODEL_COMPARISON_SCHEDULE, tracking=None, output_dir=None, verbose=True,
              n_trials=30, random_seed=42):
    """Evaluate one model independently; log a single compact completed result.

    tracking=None disables recording, useful for tests and direct experiments.
    Built-in neural functions preserve the original inner-checkpoint procedure.
    Only outer/final refits receive selected inner fit records, never test labels
    for checkpoint selection. output_dir optionally exports final predictions.
    """
    if tracking is not None:
        # Respect the compact recording contract even in a reused notebook kernel.
        import mlflow
        mlflow.autolog(disable=True)
    metadata = deepcopy(metadata or {})
    metadata['hyperparameter_search'] = dict(sampler='optuna-random', n_trials=n_trials, random_seed=random_seed)
    metadata.setdefault('data_id', data_identity(data))
    metadata.setdefault('target_id', TARGET_ID)
    metadata.setdefault('scoring_id', SCORING_ID)
    metadata.setdefault('evaluation_protocol', EVALUATION_PROTOCOL)
    metadata.setdefault('architecture_id', model_id)
    metadata.setdefault('training_protocol', 'unspecified')
    metadata.setdefault('features', [])
    metadata.setdefault('target_definition', {'column': 'winner', 'classes': list(PARTIES),
                                               'unknown_outcomes': 'exclude from accuracy'})
    metadata.setdefault('scoring_definition', 'fraction of correct labelled constituency winners; unweighted election means')
    frame = prepared_frame(data)
    missing = set(metadata['features']) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required feature columns: {sorted(missing)}')
    validate_schedule(frame, schedule)
    result = dict(model_id=model_id, schedule=deepcopy(schedule),
                  data_id=metadata['data_id'], metadata=metadata,
                  hyperparameter_candidates=deepcopy(hyperparameter_candidates),
                  outer_results={}, run_id=None)
    fit_cache = {}
    for election in schedule['outer_elections']:
        if verbose:
            print(f'Outer election {election}: tuning and fitting {model_id}', flush=True)
        history, test = split_before_election(frame, election,
                                              training_cutoff=schedule.get('training_cutoff'))
        tuned = tune_hyperparameters(model_function, hyperparameter_candidates, history,
            _inner_elections(schedule, election), training_cutoff=schedule.get('training_cutoff'),
            fixed_settings=metadata.get('fixed_settings'), verbose=verbose, _fit_cache=fit_cache,
            n_trials=n_trials, random_seed=random_seed)
        # Include unknown final outcomes in optional exports, exclude them from scores.
        if schedule['purpose'] == 'final_evaluation':
            test = data.loc[pd.to_numeric(data.election) == election].copy()
        refit_parameters = {**deepcopy(metadata.get('fixed_settings', {})),
                            **tuned['best_hyperparameters']}
        details = _evaluate(model_function, history, test, refit_parameters,
                            fit_records=tuned['fit_records'], return_details=True,
                            output_dir=output_dir if schedule['purpose'] == 'final_evaluation' else None)
        result['outer_results'][election] = dict(
            hyperparameters=tuned['best_hyperparameters'],
            mean_inner_accuracy=tuned['mean_inner_accuracy'],
            inner_accuracies=tuned['inner_accuracies'], accuracy=details['accuracy'],
            training_years=details.get('training_years', sorted(history.election.unique())),
            refit_durations=details.get('refit_durations', {}),
            evaluation_rows=details.get('evaluation_rows', int(test.winner.notna().sum())),
            excluded_rows=details.get('excluded_rows', int(test.winner.isna().sum())))
    result['mean_outer_accuracy'] = float(np.mean([v['accuracy'] for v in result['outer_results'].values()]))
    result['status'] = 'complete'
    if tracking is not None:
        from election.models.tracking import log_evaluation_result
        result['run_id'] = log_evaluation_result(result, tracking=tracking)
    return result
