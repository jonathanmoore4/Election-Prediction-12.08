"""Print completed 2024 evaluations from MLflow without running training."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


def print_specifications(client, run):
    """Print the configuration saved with this evaluation, not current defaults."""
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
        print(f'  Model type / architecture: {run.data.tags.get("architecture_id", "unknown")}')
        print(f'  Saved specifications unavailable: {error}')
        return
    for label, value in details:
        formatted = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        print(f'  {label}: {formatted}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default=os.environ.get(
        'MLFLOW_EXPERIMENT_NAME', 'election-prediction'))
    parser.add_argument('--tracking-uri', default=os.environ.get(
        'MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))
    args = parser.parse_args(argv)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        print(f'No MLflow experiment named {args.experiment!r}.')
        return
    token = None
    found = False
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
            if not found:
                print(f"{'Completed (UTC)':<20} {'Model':<24} {'2024 accuracy':>13}  Run ID")
                found = True
            completed = datetime.fromtimestamp(
                run.info.end_time / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            model = run.data.tags.get('model_id', 'unknown')
            print(f'{completed:<20} {model:<24} {accuracy:>12.2%}  {run.info.run_id}')
            print_specifications(client, run)
        token = runs.token
        if not token:
            break
    if not found:
        print('No completed 2024 evaluations recorded.')


if __name__ == '__main__':
    main()
