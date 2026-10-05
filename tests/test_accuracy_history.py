"""History displays saved tuning details and tolerates missing artifacts."""
import json
from types import SimpleNamespace

from election.reports import accuracy_history


def test_saved_specifications(tmp_path):
    artifact = tmp_path / 'evaluation.json'
    artifact.write_text(json.dumps({
        'metadata': {'architecture_id': 'random_forest-v1',
                     'training_protocol': 'full-history-v1',
                     'fixed_settings': {'random_state': 42}},
        'hyperparameter_candidates': {'max_depth': [5, 10, None]},
        'outer_results': {'2024': {'hyperparameters': {'max_depth': None}}},
    }))
    calls = []

    def download(run_id, filename, dst_path):
        calls.append((run_id, filename))
        return str(artifact)

    run = SimpleNamespace(info=SimpleNamespace(run_id='saved-run'))
    details = accuracy_history.read_specifications(SimpleNamespace(download_artifacts=download), run)
    assert calls == [('saved-run', 'evaluation.json')]
    assert details['Model type / architecture'] == 'random_forest-v1'
    assert details['Fixed settings'] == {'random_state': 42}
    assert details['Hyperparameter search space'] == {'max_depth': [5, 10, None]}
    assert details['Selected 2024 hyperparameters'] == {'max_depth': None}


def test_missing_artifact_preserves_history_output():
    def download(*args, **kwargs):
        raise FileNotFoundError('evaluation.json')

    run = SimpleNamespace(info=SimpleNamespace(run_id='old-run'),
                          data=SimpleNamespace(tags={'architecture_id': 'old-model-v1'}))
    details = accuracy_history.read_specifications(SimpleNamespace(download_artifacts=download), run)
    assert details['Model type / architecture'] == 'old-model-v1'
    assert details['Saved specifications unavailable'] == 'evaluation.json'


def test_history_collects_all_pages_and_retains_accuracy(monkeypatch):
    class Page(list):
        def __init__(self, runs, token=None):
            super().__init__(runs)
            self.token = token

    def run(run_id, metrics):
        return SimpleNamespace(
            info=SimpleNamespace(run_id=run_id, end_time=1000),
            data=SimpleNamespace(tags={'model_id': 'example'}, metrics=metrics))

    tokens = []

    def search_runs(*args, **kwargs):
        tokens.append(kwargs['page_token'])
        if kwargs['page_token'] is None:
            return Page([run('first', {'accuracy_2024': 0.75}), run('skip', {})], 'next')
        return Page([run('second', {'accuracy_2024': 0.5})])

    client = SimpleNamespace(
        get_experiment_by_name=lambda name: SimpleNamespace(experiment_id='1'),
        search_runs=search_runs)
    monkeypatch.setattr(accuracy_history, 'read_specifications',
                        lambda client, run: {'Fixed settings': {'seed': 42}})
    frame = accuracy_history.accuracy_history(client)
    assert tokens == [None, 'next']
    assert frame['Run ID'].tolist() == ['first', 'second']
    assert frame['2024 accuracy'].tolist() == [0.75, 0.5]
    assert frame.iloc[0]['Fixed settings'] == {'seed': 42}
    assert str(frame['Completed (UTC)'].dt.tz) == 'UTC'


def test_missing_experiment_returns_empty_frame():
    client = SimpleNamespace(get_experiment_by_name=lambda name: None)
    frame = accuracy_history.accuracy_history(client)
    assert frame.empty
    assert frame.columns.tolist() == accuracy_history.COLUMNS
