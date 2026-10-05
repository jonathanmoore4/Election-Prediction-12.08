"""Tabulate completed 2024 evaluations from MLflow without running training."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


def read_specifications(client, run):
    """Return the configuration saved with this evaluation, not current defaults."""
    try:
        with TemporaryDirectory() as directory:
            path = client.download_artifacts(
                run.info.run_id, 'evaluation.json', dst_path=directory)
            result = json.loads(Path(path).read_text())
        metadata = result['metadata']
        fold = result['outer_results']['2024']
        details = [
            ('Model type / architecture', metadata['architecture_id']),
            ('Training protocol', metadata['training_protocol']),
            ('Fixed settings', metadata['fixed_settings']),
            ('Hyperparameter search space', result['hyperparameter_candidates']),
            ('Selected 2024 hyperparameters', fold['hyperparameters']),
        ]
        if metadata.get('early_stopping'):
            details.append(('Early stopping', metadata['early_stopping']))
    except (MlflowException, OSError, ValueError, KeyError, TypeError) as error:
        return {
            'Model type / architecture': run.data.tags.get('architecture_id', 'unknown'),
            'Saved specifications unavailable': str(error),
        }
    return dict(details)


COLUMNS = [
    'Completed (UTC)', 'Model', '2024 accuracy', 'Run ID',
    'Model type / architecture', 'Training protocol', 'Fixed settings',
    'Hyperparameter search space', 'Selected 2024 hyperparameters',
    'Early stopping', 'Saved specifications unavailable',
]


def accuracy_history(client, experiment_name='election-prediction'):
    """Return one row per completed evaluation, with accuracy as a proportion.

    Configuration columns retain dictionaries/lists for further analysis.
    An absent experiment or an experiment without evaluations returns an empty
    DataFrame with the same columns.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame(columns=COLUMNS)
    token = None
    rows = []
    while True:
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string="tags.purpose = 'final_evaluation' AND "
                          "tags.record_complete = 'true' AND attributes.status = 'FINISHED'",
            order_by=['attributes.end_time DESC'],
            max_results=1000, page_token=token,
        )
        for run in runs:
            accuracy = run.data.metrics.get('accuracy_2024')
            if accuracy is None:
                continue
            rows.append({
                'Completed (UTC)': datetime.fromtimestamp(
                    run.info.end_time / 1000, timezone.utc),
                'Model': run.data.tags.get('model_id', 'unknown'),
                '2024 accuracy': accuracy,
                'Run ID': run.info.run_id,
                **read_specifications(client, run),
            })
        token = runs.token
        if not token:
            break
    return pd.DataFrame(rows, columns=COLUMNS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default=os.environ.get(
        'MLFLOW_EXPERIMENT_NAME', 'election-prediction'))
    parser.add_argument('--tracking-uri', default=os.environ.get(
        'MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))
    parser.add_argument('--full', action='store_true',
                        help='Show full configuration cells instead of shortening them.')
    args = parser.parse_args(argv)
    frame = accuracy_history(MlflowClient(tracking_uri=args.tracking_uri), args.experiment)
    if frame.empty:
        print(f'No completed 2024 evaluations recorded in {args.experiment!r}.')
    else:
        display = frame.copy()
        for column in display.columns:
            display[column] = display[column].map(
                lambda value: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list)) else value)
        display = display.fillna('')
        print(display.to_string(
            index=False, max_colwidth=None if args.full else 60,
            formatters={'2024 accuracy': lambda value: f'{value:.2%}'},
        ))
    return frame


if __name__ == '__main__':
    main()
