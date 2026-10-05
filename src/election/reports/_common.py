"""Shared formatting and connection options for terminal reports."""
import json
import os


def tracking_options(parser):
    parser.add_argument('--experiment', default=os.environ.get(
        'MLFLOW_EXPERIMENT_NAME', 'election-prediction'))
    parser.add_argument('--tracking-uri', default=os.environ.get(
        'MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))


def tracking_settings(args):
    return dict(experiment_name=args.experiment, tracking_uri=args.tracking_uri)


def print_fields(fields):
    for label, value in fields:
        if isinstance(value, (dict, list)):
            value = json.dumps(value, sort_keys=True)
        print(f'{label}: {value}')
