"""Print completed 2024 evaluations from MLflow without running training."""
import argparse
from datetime import datetime, timezone
import os

from mlflow import MlflowClient


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
        token = runs.token
        if not token:
            break
    if not found:
        print('No completed 2024 evaluations recorded.')


if __name__ == '__main__':
    main()
