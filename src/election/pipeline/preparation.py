"""Independent preparation and immutable prepared-data locations."""
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path

import pandas as pd

from election import paths
from election.models.evaluation import data_identity
from election.pipeline.helpers import read_in_raw, clean_data_all, predictor_guide
from election.sql import apply_sql_queries


def _model_frame(frame):
    """Preserve the historical model dtypes and content IDs without CSV files."""
    return pd.read_csv(StringIO(frame.to_csv(index=False)))


def run_preparation(config=None):
    """Run existing ingestion, cleaning and SQL once; do not start training."""
    config = config or {}
    output = Path(config.get('output_dir', paths.project_root() / 'outputs')).resolve()
    raw = read_in_raw.read_raw_data()
    cleaned = clean_data_all.clean_all_data(raw)
    frames = apply_sql_queries.apply_sql_queries(cleaned)
    # Use the same model representation for preparation and database reads.
    train = _model_frame(frames['train'])
    test = _model_frame(frames['test'])
    data_id, test_id = data_identity(train), data_identity(test)
    snapshot_id = sha256((data_id + test_id).encode()).hexdigest()
    directory = output / 'datasets' / snapshot_id
    directory.mkdir(parents=True, exist_ok=True)
    prepared = dict(data_id=data_id, test_data_id=test_id,
        predictor_guide_path=str(directory / 'predictor_descriptions.md'),
        database_schema=frames['train'].attrs.get('database_schema'))
    guide = Path(prepared['predictor_guide_path'])
    if not guide.exists():
        predictor_guide.write_predictor_guide(frames, guide)
    manifest = directory / 'prepared_data.json'
    if not manifest.exists():
        manifest.write_text(json.dumps(prepared, indent=2) + '\n')
    else:
        # Preserve the original retained SQL schema reference for reused snapshots.
        prepared = json.loads(manifest.read_text())
    # Verify retained tables before publishing the latest snapshot pointer.
    # Changed snapshots must fail rather than be silently replaced.
    load_prepared(prepared, include_test=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'latest_prepared.json').write_text(json.dumps(prepared, indent=2) + '\n')
    return prepared


def prepared_locations(value):
    """Accept a preparation dictionary, manifest path or snapshot directory."""
    if isinstance(value, dict):
        return dict(value)
    path = Path(value).resolve()
    if path.is_dir():
        snapshot = path / 'prepared_data.json'
        path = snapshot if snapshot.exists() else path / 'latest_prepared.json'
    result = json.loads(path.read_text())
    for key in ('train_path', 'test_path', 'predictor_guide_path'):
        if key in result and not Path(result[key]).is_absolute():
            result[key] = str((path.parent / result[key]).resolve())
    return result


def load_prepared(value, *, include_test=False):
    locations = prepared_locations(value)
    if locations.get('database_schema'):
        frames = apply_sql_queries.read_snapshot(
            locations['database_schema'], include_test=include_test)
        train = _model_frame(frames['train'])
    else:
        # Explicit CSV inputs remain supported for external datasets.
        train = pd.read_csv(locations['train_path'])
    identity = data_identity(train)
    if locations.get('data_id', identity) != identity:
        raise ValueError('Prepared training data has changed since preparation.')
    locations['data_id'] = identity
    if not include_test:
        return train, locations
    test = (_model_frame(frames['test']) if locations.get('database_schema')
            else pd.read_csv(locations['test_path']))
    test_id = data_identity(test)
    if locations.get('test_data_id', test_id) != test_id:
        raise ValueError('Prepared final test data has changed since preparation.')
    if (pd.to_numeric(train.election) > 2019).any() or not (pd.to_numeric(test.election) == 2024).all():
        raise ValueError('Prepared history must end by 2019; final test must contain only 2024.')
    locations['test_data_id'] = test_id
    return pd.concat([train, test], ignore_index=True), locations
