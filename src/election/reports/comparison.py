"""Print saved historical rankings, selected model and selection run ID."""
import argparse
from pathlib import Path

from election import paths
from election.pipeline.run_pipeline import MODEL_MODULES, compare_recorded
from election.reports._common import tracking_options, tracking_settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=paths.project_root() / 'outputs' / 'latest_prepared.json')
    parser.add_argument('--model', nargs='+', choices=list(MODEL_MODULES))
    tracking_options(parser)
    args = parser.parse_args(argv)
    result = compare_recorded(args.data_dir, model_ids=args.model,
                              tracking=tracking_settings(args))
    print(f'{"Model":<24} {"Historical accuracy":>19}  Run ID')
    for row in result['ranked_results']:
        print(f'{row["model_id"]:<24} {row["mean_outer_accuracy"]:>18.2%}  {row["run_id"]}')
    winner = result['winner_model_id']
    if result['missing_results']:
        print('Comparison incomplete; ranking covers available compatible results.')
    if winner:
        print(f'Highest historical accuracy: {winner}')
        print(f'Selection run ID: {result["source_run_ids"][winner]}')
    else:
        print('No completed compatible historical evaluations found.')
    for row in result['missing_results']:
        print(f'Missing {row["model_id"]}: {row["reason"]}')
    for row in result['excluded_results']:
        print(f'Excluded {row["model_id"]} ({row["run_id"]}): {row["reason"]}')


if __name__ == '__main__':
    main()
