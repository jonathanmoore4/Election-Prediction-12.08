"""Print one completed MLflow evaluation's settings and election scores."""
import argparse

from election.models.tracking import read_evaluation
from election.reports._common import print_fields, tracking_options, tracking_settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id', help='Completed historical or final MLflow run ID')
    tracking_options(parser)
    args = parser.parse_args(argv)
    result = read_evaluation(args.run_id, tracking=tracking_settings(args))
    metadata = result['metadata']
    print_fields([
        ('Run ID', result['run_id']), ('Model', result['model_id']),
        ('Schedule', result['schedule']),
        ('Mean election accuracy', f'{result["mean_outer_accuracy"]:.2%}'),
        ('Model type / architecture', metadata['architecture_id']),
        ('Training protocol', metadata['training_protocol']),
        ('Fixed settings', metadata.get('fixed_settings', {})),
        ('Hyperparameter search space', result['hyperparameter_candidates']),
    ])
    for key in ('early_stopping', 'features', 'data_id', 'prepared_data',
                'selection_run_id', 'source_run_ids'):
        if key in metadata:
            print_fields([(key.replace('_', ' ').capitalize(), metadata[key])])
    for year, fold in sorted(result['outer_results'].items(), key=lambda item: int(item[0])):
        print(f'\nElection {year}: {fold["accuracy"]:.2%} accuracy')
        print_fields([
            ('  Selected hyperparameters', fold['hyperparameters']),
            ('  Mean inner accuracy', f'{fold["mean_inner_accuracy"]:.2%}'),
            ('  Inner election accuracies', {year: f'{score:.2%}'
                for year, score in fold['inner_accuracies'].items()}),
            ('  Training years', fold['training_years']),
        ])
        for key in ('refit_durations', 'evaluation_rows', 'excluded_rows'):
            if key in fold:
                print_fields([('  ' + key.replace('_', ' ').capitalize(), fold[key])])


if __name__ == '__main__':
    main()
