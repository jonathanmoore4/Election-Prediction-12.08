"""Shared scoring and optional local diagnostics, without experiment logging."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from election.models import custom_model
from election.models.config import PARTIES


def evaluate_fit(factory, train_data, test_data, hyperparameters, *, metadata,
                 context=None, return_details=False, output_dir=None):
    """Keep fitted implementation objects inside one call; return numbers/dicts."""
    unknown = set(hyperparameters) - set(metadata['supported_hyperparameters'])
    if unknown:
        raise ValueError(f"{metadata['model_id']}: unsupported hyperparameters {sorted(unknown)}")
    features = metadata['features']
    training = train_data[list(dict.fromkeys([*features, 'election', 'winner']))].copy()
    test = test_data[list(dict.fromkeys([*features, 'election']))].copy()
    labelled = test_data.winner.notna().to_numpy()
    if not labelled.any():
        raise ValueError('Evaluation requires known winners.')
    if set(test_data.loc[labelled, 'winner']) - set(PARTIES):
        raise ValueError('Observed party is outside the declared class convention.')
    context = context if context is not None else custom_model.fit_context()
    if context['validation'] is not None:
        context = {**context, 'validation': context['validation'][training.columns].copy()}
    model = factory()
    record = model.train(training, {**deepcopy(metadata['fixed_settings']),
                                    **deepcopy(hyperparameters)}, context)
    probabilities = np.asarray(model.predict_proba(test), dtype=float)
    if (tuple(model.classes_) != PARTIES or probabilities.shape != (len(test), len(PARTIES))
            or not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)):
        raise ValueError('Model returned invalid or misaligned probabilities.')
    predicted = np.asarray(PARTIES)[probabilities.argmax(axis=1)]
    accuracy = float((predicted[labelled] == test_data.loc[labelled, 'winner']).mean())
    details = dict(accuracy=accuracy, fit_record=record,
                   hyperparameters=deepcopy(model.hyperparameters),
                   training_years=record['training_years'],
                   refit_durations=deepcopy(context['durations']),
                   evaluation_rows=int(labelled.sum()), excluded_rows=int((~labelled).sum()))
    if output_dir is not None:
        export_predictions(test_data, predicted, output_dir, metadata['model_id'])
    return details if return_details else accuracy


def export_predictions(data, predicted, output_dir, model_id):
    """Optional constituency reports stay on disk, never in MLflow."""
    from matplotlib.figure import Figure
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    export = data.copy()
    export['predicted_winner'] = predicted
    export.to_csv(output / 'test_predictions.csv', index=False)
    labelled = data.winner.notna().to_numpy()
    labels = sorted(set(data.loc[labelled, 'winner']) | set(predicted[labelled]))
    matrix = confusion_matrix(data.loc[labelled, 'winner'], predicted[labelled], labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output / 'test_confusion_matrix.csv', index_label='actual_winner')
    figure = Figure(figsize=(8, 6))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
        ax=figure.subplots(), cmap='Blues', values_format='d', colorbar=False)
    figure.axes[0].set_title(f'{model_id}: {int(data.election.iloc[0])}')
    figure.tight_layout()
    figure.savefig(output / 'test_confusion_matrix.png', dpi=150, bbox_inches='tight')
