"""Compare recorded historical evaluations without loading data or training."""
from election.models.tracking import latest_completed_evaluations


def compare_models(*, model_ids, compatibility, tracking):
    records = latest_completed_evaluations(
        model_ids=model_ids, compatibility=compatibility, tracking=tracking)
    order = {name: i for i, name in enumerate(model_ids)}
    ranked = sorted(records['evaluations'].values(),
                    key=lambda row: (-row['mean_outer_accuracy'], order[row['model_id']]))
    return dict(
        ranked_results=[dict(model_id=row['model_id'], mean_outer_accuracy=row['mean_outer_accuracy'],
                             run_id=row['run_id'], completed_at=row['completed_at']) for row in ranked],
        winner_model_id=ranked[0]['model_id'] if ranked else None,
        source_run_ids={row['model_id']: row['run_id'] for row in ranked},
        compatibility=dict(compatibility),
        missing_results=records['missing_results'], excluded_results=records['excluded_results'])
