"""MLflow execution groups, selection snapshots and compact evaluation records."""
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from contextlib import contextmanager
from functools import wraps
from tempfile import TemporaryDirectory

from mlflow import MlflowClient

from election.models.config import MODEL_COMPARISON_SCHEDULE
from election.models.evaluation import json_value


def lineage_tags(tracking):
    execution_id = (tracking or {}).get('execution_id')
    return {'execution_id': execution_id, 'mlflow.parentRunId': execution_id} if execution_id else {}


@contextmanager
def execution_scope(execution_type, tracking):
    """Group one invocation; reused evaluations stay under their original parents."""
    settings = dict(tracking or {})
    if settings.get('execution_id'):
        yield settings
        return
    client, experiment_id = _client(settings)
    run_id = client.create_run(experiment_id, tags={
        'mlflow.runName': f'Execution: {execution_type}',
        'record_type': 'execution-v1', 'run_role': 'execution',
        'execution_type': execution_type,
    }).info.run_id
    settings['execution_id'] = run_id
    try:
        yield settings
        client.set_terminated(run_id, 'FINISHED')
    except BaseException as error:
        try:
            client.set_terminated(run_id, 'FAILED')
        except Exception as logging_error:
            error.add_note(f'Unable to mark failed execution: {logging_error}')
        raise
    _refresh_result_status(client, experiment_id, {
        'record_type': 'execution-v1', 'execution_type': execution_type})


def tracked_execution(execution_type):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if kwargs.get('record') is False:
                return function(*args, **kwargs)
            with execution_scope(execution_type, kwargs.get('tracking')) as settings:
                kwargs['tracking'] = settings
                return function(*args, **kwargs)
        return wrapped
    return decorate


def log_stage_result(result, *, role, tracking):
    """Save a preparation or comparison snapshot and refresh its status group."""
    client, experiment_id = _client(tracking)
    run_id = client.create_run(experiment_id, tags={
        'mlflow.runName': role, 'record_type': f'{role}-v1',
        'run_role': role, 'record_complete': 'false', **lineage_tags(tracking),
    }).info.run_id
    try:
        value = {**json_value(result), 'run_id': run_id}
        client.log_dict(run_id, value, f'{role}.json')
        if role == 'selection' and result.get('winner_model_id'):
            client.set_tag(run_id, 'winner_model_id', result['winner_model_id'])
            client.set_tag(run_id, 'selected_evaluation_run_id', result['source_run_ids'][result['winner_model_id']])
        client.set_tag(run_id, 'record_complete', 'true')
        client.set_terminated(run_id, 'FINISHED')
    except BaseException as error:
        try:
            client.set_tag(run_id, 'record_complete', 'false')
            client.set_terminated(run_id, 'FAILED')
        except Exception as logging_error:
            error.add_note(f'Unable to mark failed stage: {logging_error}')
        raise
    _refresh_result_status(client, experiment_id, {'record_type': f'{role}-v1'})
    return run_id


def _client(tracking):
    settings = tracking or {}
    uri = settings.get('tracking_uri') or os.environ.get('MLFLOW_TRACKING_URI') or 'http://127.0.0.1:5000'
    if not uri.startswith(('http://', 'https://')):
        raise ValueError('Use the existing PostgreSQL-backed MLflow tracking server.')
    client = MlflowClient(tracking_uri=uri)
    name = settings.get('experiment_name') or os.environ.get('MLFLOW_EXPERIMENT_NAME', 'election-prediction')
    experiment = client.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(name)
    return client, experiment_id


def schedule_identity(schedule):
    return sha256(json.dumps(json_value(schedule), sort_keys=True, allow_nan=False).encode()).hexdigest()


def compatibility_fields(metadata, schedule=MODEL_COMPARISON_SCHEDULE):
    """Match data, full schedule, target, scoring and the evaluation procedure."""
    return dict(data_id=metadata['data_id'], schedule_id=schedule_identity(schedule),
                target_id=metadata['target_id'], scoring_id=metadata['scoring_id'],
                evaluation_protocol=metadata['evaluation_protocol'])


def _valid_accuracy(value):
    return (not isinstance(value, bool) and isinstance(value, (float, int))
            and math.isfinite(value) and 0 <= value <= 1)


def validate_evaluation(result):
    expected = {str(year) for year in result['schedule']['outer_elections']}
    actual = {str(year) for year in result['outer_results']}
    if not expected or actual != expected or result.get('status') != 'complete':
        raise ValueError('Incomplete evaluation: every scheduled outer election is required.')
    scores = []
    for year, fold in result['outer_results'].items():
        if not _valid_accuracy(fold.get('accuracy')) or not _valid_accuracy(fold.get('mean_inner_accuracy')):
            raise ValueError(f'Invalid accuracy for election {year}.')
        mapping = result['schedule']['inner_elections']
        inner = mapping.get(int(year), mapping.get(str(year), []))
        if {str(v) for v in fold['inner_accuracies']} != {str(v) for v in inner}:
            raise ValueError(f'Incomplete inner summary for election {year}.')
        inner_scores = list(fold['inner_accuracies'].values())
        if not inner_scores or not all(_valid_accuracy(v) for v in inner_scores):
            raise ValueError(f'Invalid inner accuracy for election {year}.')
        if not math.isclose(sum(inner_scores) / len(inner_scores), fold['mean_inner_accuracy'], abs_tol=1e-12):
            raise ValueError('Inner mean must give every election equal weight.')
        history = fold.get('training_years', [])
        cutoff = result['schedule'].get('training_cutoff')
        if not history or any(v >= int(year) or (cutoff is not None and v > cutoff) for v in history):
            raise ValueError('Recorded training history violates election boundaries.')
        scores.append(fold['accuracy'])
    mean = result.get('mean_outer_accuracy')
    if not _valid_accuracy(mean) or not math.isclose(mean, sum(scores) / len(scores), abs_tol=1e-12):
        raise ValueError('Outer mean must give every election equal weight.')


def _refresh_result_status(client, experiment_id, tags):
    """Label latest successful results independently of comparison compatibility."""
    record_type = tags['record_type']
    keys = ('record_type',)
    if record_type == 'evaluation-v1':
        keys += ('model_id', 'purpose')
    elif record_type == 'execution-v1':
        keys += ('execution_type',)
    runs, token = [], None
    while True:
        page = client.search_runs([experiment_id],
            filter_string=f"tags.record_type = '{record_type}'",
            order_by=['attributes.end_time DESC', 'attributes.run_id ASC'],
            max_results=1000, page_token=token)
        runs.extend(run for run in page
                    if run.info.status == 'FINISHED'
                    and run.info.end_time is not None
                    and (record_type == 'execution-v1'
                         or run.data.tags.get('record_complete') == 'true')
                    and all(run.data.tags.get(key) == tags[key] for key in keys))
        token = page.token
        if not token:
            break
    for index, run in enumerate(runs):
        status = 'Current' if index == 0 else 'Old'
        if run.data.tags.get('result_status') != status:
            client.set_tag(run.info.run_id, 'result_status', status)


def log_evaluation_result(result, *, tracking, fitted_model=None, confusion_matrix_figure=None):
    """Record compact summaries and, for final evaluation only, the fitted model."""
    value = json_value(result)
    validate_evaluation(value)
    client, experiment_id = _client(tracking)
    metadata = value['metadata']
    tags = {
        'mlflow.runName': f"{value['model_id']}: {value['schedule']['name']}",
        'record_type': 'evaluation-v1', 'record_complete': 'false',
        'model_id': value['model_id'], 'architecture_id': metadata['architecture_id'],
        'training_protocol': metadata['training_protocol'],
        'purpose': value['schedule']['purpose'], 'schedule_name': value['schedule']['name'],
        **compatibility_fields(metadata, value['schedule']),
        'run_role': 'final_evaluation' if value['schedule']['purpose'] == 'final_evaluation' else 'historical_evaluation',
        **lineage_tags(tracking),
    }
    if metadata.get('selection_run_id'):
        tags['selection_run_id'] = metadata['selection_run_id']
    if metadata.get('comparison_run_id'):
        tags['comparison_run_id'] = metadata['comparison_run_id']
    run_id = client.create_run(experiment_id, tags=tags).info.run_id
    logged_model_id = None
    try:
        # An explicit whitelist prevents internal fit records or caller dataframes
        # from becoming accidental large artifacts.
        fields = ('model_id', 'schedule', 'data_id', 'hyperparameter_candidates',
                  'outer_results', 'mean_outer_accuracy', 'status')
        artifact = {key: value[key] for key in fields}
        metadata_fields = ('hyperparameter_search', 'architecture_id', 'training_protocol', 'fixed_settings',
            'early_stopping', 'features', 'feature_definition_reference',
            'target_id', 'target_definition', 'scoring_id', 'scoring_definition', 'evaluation_protocol', 'data_id',
            'prepared_data', 'selection_run_id', 'comparison_run_id', 'source_run_ids')
        artifact['metadata'] = {key: metadata[key] for key in metadata_fields if key in metadata}
        artifact['run_id'] = run_id
        if fitted_model is not None:
            if value['schedule']['purpose'] != 'final_evaluation':
                raise ValueError('Only the final fitted model is saved.')
            from election.models.saved_model import save_model
            logged_model_id = client.create_logged_model(
                experiment_id, name=f"{value['model_id']}-2024", source_run_id=run_id,
                model_type='election-winner', tags={'model_id': value['model_id']}).model_id
            with TemporaryDirectory(prefix='election-final-model-') as directory:
                model_path = Path(directory) / 'model'
                save_model(fitted_model, model_path, model_id=logged_model_id, run_id=run_id)
                client.log_model_artifacts(logged_model_id, str(model_path))
            final_parameters = {**metadata.get('fixed_settings', {}),
                                **value['outer_results']['2024']['hyperparameters']}
            client.log_model_params(logged_model_id, {
                key: json.dumps(setting, sort_keys=True) for key, setting in final_parameters.items()})
            artifact['model_uri'] = f'models:/{logged_model_id}'
            client.set_tag(run_id, 'model_uri', artifact['model_uri'])
        if confusion_matrix_figure is not None:
            if value['schedule']['purpose'] != 'final_evaluation':
                raise ValueError('Only final evaluations log confusion matrix figures.')
            client.log_figure(run_id, confusion_matrix_figure, 'test_confusion_matrix.png',
                              save_kwargs={'dpi': 150, 'bbox_inches': 'tight'})
        client.log_dict(run_id, artifact, 'evaluation.json')
        metric_options = {'model_id': logged_model_id} if logged_model_id else {}
        client.log_metric(run_id, 'mean_outer_accuracy', value['mean_outer_accuracy'], **metric_options)
        for year, fold in value['outer_results'].items():
            client.log_metric(run_id, f'accuracy_{year}', fold['accuracy'], **metric_options)
        cutoff = value['schedule'].get('training_cutoff')
        if cutoff is not None:
            client.log_param(run_id, 'training_cutoff', cutoff)
        for key in ('random_state', 'seeds', 'max_epochs', 'patience'):
            if key in metadata.get('fixed_settings', {}):
                setting = metadata['fixed_settings'][key]
                client.log_param(run_id, key, json.dumps(setting) if isinstance(setting, list) else setting)
        if logged_model_id is not None:
            client.finalize_logged_model(logged_model_id, 'READY')
        client.set_tag(run_id, 'record_complete', 'true')
        client.set_terminated(run_id, 'FINISHED')
    except BaseException as error:
        try:
            if logged_model_id is not None:
                client.finalize_logged_model(logged_model_id, 'FAILED')
            client.set_tag(run_id, 'record_complete', 'false')
            client.set_terminated(run_id, 'FAILED')
        except Exception as logging_error:
            error.add_note(f'Unable to mark failed MLflow run: {logging_error}')
        raise
    _refresh_result_status(client, experiment_id, tags)
    return run_id


def _read_completed(client, run):
    if run.info.status != 'FINISHED' or run.data.tags.get('record_complete') != 'true' or run.info.end_time is None:
        raise ValueError('Run is not a completed evaluation.')
    artifact_path = client.download_artifacts(run.info.run_id, 'evaluation.json')
    result = json.loads(Path(artifact_path).read_text())
    validate_evaluation(result)
    if result['run_id'] != run.info.run_id or result['model_id'] != run.data.tags.get('model_id'):
        raise ValueError('Artifact identity does not match the MLflow run.')
    fields = compatibility_fields(result['metadata'], result['schedule'])
    if any(run.data.tags.get(key) != value for key, value in fields.items()):
        raise ValueError('Artifact compatibility does not match MLflow tags.')
    if run.data.tags.get('purpose') != result['schedule']['purpose']:
        raise ValueError('Artifact purpose does not match MLflow tags.')
    expected_metrics = {'mean_outer_accuracy': result['mean_outer_accuracy'],
                        **{f'accuracy_{year}': fold['accuracy'] for year, fold in result['outer_results'].items()}}
    if any(not _valid_accuracy(run.data.metrics.get(key)) or
           not math.isclose(run.data.metrics[key], value, abs_tol=1e-12)
           for key, value in expected_metrics.items()):
        raise ValueError('MLflow metrics do not match the completed summary.')
    result['completed_at'] = run.info.end_time
    return result


def read_evaluation(run_id, *, tracking):
    client, _ = _client(tracking)
    return _read_completed(client, client.get_run(run_id))


def latest_completed_evaluations(*, model_ids, compatibility, tracking):
    """Read newest compatible historical run per model; never select by score."""
    if not model_ids or len(set(model_ids)) != len(model_ids):
        raise ValueError('Provide nonempty unique model identifiers.')
    required = {'data_id', 'schedule_id', 'target_id', 'scoring_id', 'evaluation_protocol'}
    if set(compatibility) != required:
        raise ValueError(f'Compatibility must specify {sorted(required)}.')
    client, experiment_id = _client(tracking)
    results, excluded, token = {}, [], None
    while True:
        page = client.search_runs([experiment_id],
            filter_string="tags.record_type = 'evaluation-v1'",
            order_by=['attributes.end_time DESC', 'attributes.run_id ASC'],
            max_results=1000, page_token=token)
        for run in page:
            model_id = run.data.tags.get('model_id')
            if model_id not in model_ids or model_id in results:
                continue
            reason = None
            if run.data.tags.get('purpose') != 'model_comparison':
                reason = 'Final evaluations are excluded from model comparison.'
            elif run.info.status != 'FINISHED' or run.data.tags.get('record_complete') != 'true':
                reason = 'Evaluation is incomplete or failed.'
            else:
                mismatches = [key for key, expected in compatibility.items()
                              if run.data.tags.get(key) != expected]
                if mismatches:
                    reason = 'Incompatible ' + ', '.join(mismatches) + '.'
            if reason is None:
                try:
                    results[model_id] = _read_completed(client, run)
                except (ValueError, KeyError, TypeError) as error:
                    reason = f'Invalid completed summary: {error}'
            if reason:
                excluded.append(dict(model_id=model_id, run_id=run.info.run_id, reason=reason))
        token = page.token
        if not token or len(results) == len(model_ids):
            break
    missing = [dict(model_id=model_id, reason='No completed compatible historical evaluation found.')
               for model_id in model_ids if model_id not in results]
    return dict(evaluations=results, excluded_results=excluded, missing_results=missing)
