"""Independent stage functions and one lightweight pipeline/CLI entry point."""
import argparse
from copy import deepcopy
from importlib import import_module
import json
from pathlib import Path

from election import paths
from election.models.config import (
    MODEL_COMPARISON_SCHEDULE, FINAL_EVALUATION_SCHEDULE,
    TARGET_ID, SCORING_ID, EVALUATION_PROTOCOL,
)
from election.models.evaluation import nested_cv, json_value
from election.models.compare_models import compare_models
from election.models.tracking import compatibility_fields, read_evaluation
from election.pipeline.preparation import run_preparation, load_prepared, prepared_locations

PROJECT_ROOT = paths.project_root()

# Only wiring: functions, candidate dictionaries and metadata live in each model
# module. Existing candidate order also determines exact model-selection ties.
MODEL_MODULES = {
    'xgboost': ('xgboost_model', 'XGBoost', 'xgboosthyperparameters'),
    'random_forest': ('random_forest', 'RandomForest', 'randomforesthyperparameters'),
    'logistic_regression': ('logistic_regression', 'LogReg', 'logreghyperparameters'),
    'nn01': ('nn01_model', 'NN01', 'nn01hyperparameters'),
    'nn02': ('nn02_model', 'NN02', 'nn02hyperparameters'),
    'nn03': ('nn03_model', 'NN03', 'nn03hyperparameters'),
    'xgboost_expanded': ('xgboost_expanded_model', 'XGBoostExpanded', 'xgboostexpandedhyperparameters'),
    'conditional_xgboost': ('conditional_xgboost_model', 'ConditionalXGBoost', 'conditionalxgboosthyperparameters'),
}


def model_functions(model_id):
    try:
        module_name, function_name, candidates_name = MODEL_MODULES[model_id]
    except KeyError:
        raise ValueError(f'Unknown model {model_id!r}; choose from {list(MODEL_MODULES)}') from None
    module = import_module(f'election.models.adapters.{module_name}')
    return getattr(module, function_name), deepcopy(getattr(module, candidates_name)), deepcopy(module.TRAINING_METADATA)


def _model_ids(model_ids):
    ids = list(MODEL_MODULES) if model_ids is None else list(model_ids)
    if not ids or len(set(ids)) != len(ids) or set(ids) - set(MODEL_MODULES):
        raise ValueError(f'Provide nonempty unique model identifiers from {list(MODEL_MODULES)}.')
    return ids


def _data_metadata(locations):
    if not locations.get('data_id'):
        raise ValueError('Comparison requires a data_id; use a prepared manifest or evaluate the data first.')
    return dict(data_id=locations['data_id'], prepared_data=locations,
                feature_definition_reference=locations.get('predictor_guide_path'),
                target_id=TARGET_ID, scoring_id=SCORING_ID,
                evaluation_protocol=EVALUATION_PROTOCOL)


def evaluate_models(prepared_data, *, model_ids=None, tracking=None, verbose=True):
    """Load existing history and evaluate one or multiple models independently."""
    ids = _model_ids(model_ids)
    data, locations = load_prepared(prepared_data)
    tracking = {} if tracking is None else tracking
    results = {}
    for model_id in ids:
        function, candidates, metadata = model_functions(model_id)
        results[model_id] = nested_cv(function, candidates, data, model_id=model_id,
            metadata={**metadata, **_data_metadata(locations)}, tracking=tracking, verbose=verbose)
    return results


def compare_recorded(prepared_data, *, model_ids=None, tracking=None):
    """Read the manifest and MLflow only; never load rows or start training."""
    locations = prepared_locations(prepared_data)
    return compare_models(model_ids=_model_ids(model_ids),
        compatibility=compatibility_fields(_data_metadata(locations)),
        tracking={} if tracking is None else tracking)


def evaluate_final(prepared_data, *, selection_run_id, source_run_ids=None,
                   tracking=None, output_dir=None, verbose=True):
    """Tune/refit the historically selected model through 2019, then score 2024."""
    tracking = {} if tracking is None else tracking
    # Load outcomes only after a historical run identifies the chosen model.
    selected = read_evaluation(selection_run_id, tracking=tracking)
    if selected['schedule'] != json_value(MODEL_COMPARISON_SCHEDULE):
        raise ValueError('Selection must reference a completed five-election historical evaluation.')
    data, locations = load_prepared(prepared_data, include_test=True)
    if compatibility_fields(selected['metadata'], selected['schedule']) != compatibility_fields(_data_metadata(locations)):
        raise ValueError('Selected historical evaluation is incompatible with prepared data.')
    model_id = selected['model_id']
    function, _, metadata = model_functions(model_id)
    # Reuse the evaluated search space/fixed settings rather than silently tuning
    # a newer grid. Reject an implementation/protocol change before final fitting.
    for key in ('architecture_id', 'training_protocol', 'features'):
        if selected['metadata'][key] != metadata[key]:
            raise ValueError(f'Selected model has an incompatible {key}; rerun historical evaluation.')
    metadata['fixed_settings'] = deepcopy(selected['metadata'].get('fixed_settings', {}))
    if 'early_stopping' in selected['metadata']:
        metadata['early_stopping'] = deepcopy(selected['metadata']['early_stopping'])
    result = nested_cv(function, selected['hyperparameter_candidates'], data,
        model_id=model_id, metadata={**metadata, **_data_metadata(locations),
            'selection_run_id': selection_run_id,
            'source_run_ids': source_run_ids or {model_id: selection_run_id}},
        schedule=FINAL_EVALUATION_SCHEDULE, tracking=tracking,
        output_dir=Path(output_dir) / 'reports' if output_dir is not None else None,
        verbose=verbose)
    return result


def run_pipeline(config=None, *, output_dir=None, model_ids=None, prepared_data=None,
                 reuse_evaluations=False, final_evaluation=True, tracking=None, verbose=True):
    """Run the same independent stages, optionally reusing preparation/results.

    Returns ordinary dictionaries; fitted models remain local to individual fits.
    Preparation runs only when prepared_data is omitted. reuse_evaluations=True
    compares existing runs and never starts historical training.
    """
    config = dict(config or {})
    if output_dir is not None:
        config['output_dir'] = output_dir
    output = Path(config.get('output_dir', PROJECT_ROOT / 'outputs')).resolve()
    ids = _model_ids(model_ids)
    tracking = {} if tracking is None else tracking
    prepared = run_preparation({**config, 'output_dir': output}) if prepared_data is None else prepared_locations(prepared_data)
    if not prepared.get('data_id'):
        _, prepared = load_prepared(prepared)
    evaluations = {} if reuse_evaluations else evaluate_models(
        prepared, model_ids=ids, tracking=tracking, verbose=verbose)
    comparison = compare_recorded(prepared, model_ids=ids, tracking=tracking)
    if comparison['missing_results']:
        raise ValueError(f"Comparison requires all requested models: {comparison['missing_results']}")
    final = None
    if final_evaluation:
        winner = comparison['winner_model_id']
        final = evaluate_final(prepared, selection_run_id=comparison['source_run_ids'][winner],
            source_run_ids=comparison['source_run_ids'], tracking=tracking,
            output_dir=output, verbose=verbose)
    return dict(prepared_data=prepared, evaluations=evaluations, comparison=comparison,
                final_evaluation=final, status='complete')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'evaluate', 'compare', 'final', 'all'])
    parser.add_argument('--model', nargs='+', choices=['all', *MODEL_MODULES], default=['all'])
    parser.add_argument('--data-dir', type=Path, help='Prepared manifest, snapshot folder, or outputs folder')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'outputs')
    parser.add_argument('--selection-run', help='Completed historical MLflow run for the selected model')
    parser.add_argument('--reuse-evaluations', action='store_true')
    parser.add_argument('--skip-final', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)
    if 'all' in args.model and args.model != ['all']:
        parser.error('--model all cannot be combined with specific model names')
    model_ids = None if args.model == ['all'] else args.model
    prepared = args.data_dir or args.output_dir / 'latest_prepared.json'
    if args.stage == 'prepare':
        result = run_preparation({'output_dir': args.output_dir})
    elif args.stage == 'evaluate':
        result = evaluate_models(prepared, model_ids=model_ids, verbose=not args.quiet)
    elif args.stage == 'compare':
        result = compare_recorded(prepared, model_ids=model_ids)
    elif args.stage == 'final':
        if not args.selection_run:
            parser.error('final requires --selection-run with a completed historical evaluation')
        result = evaluate_final(prepared, selection_run_id=args.selection_run,
                                output_dir=args.output_dir, verbose=not args.quiet)
    else:
        result = run_pipeline(output_dir=args.output_dir, model_ids=model_ids,
            prepared_data=args.data_dir, reuse_evaluations=args.reuse_evaluations,
            final_evaluation=not args.skip_final, verbose=not args.quiet)
    print(json.dumps(json_value(result), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
