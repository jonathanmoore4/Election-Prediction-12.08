"""History displays saved tuning details and tolerates missing artifacts."""
import json
from types import SimpleNamespace

from election.reports import accuracy_history


def test_saved_specifications(tmp_path, capsys):
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
    accuracy_history.print_specifications(SimpleNamespace(download_artifacts=download), run)
    output = capsys.readouterr().out
    assert calls == [('saved-run', 'evaluation.json')]
    assert 'Model type / architecture: random_forest-v1' in output
    assert 'Fixed settings: {"random_state": 42}' in output
    assert 'Hyperparameter search space: {"max_depth": [5, 10, null]}' in output
    assert 'Selected 2024 hyperparameters: {"max_depth": null}' in output


def test_missing_artifact_preserves_history_output(capsys):
    def download(*args, **kwargs):
        raise FileNotFoundError('evaluation.json')

    run = SimpleNamespace(info=SimpleNamespace(run_id='old-run'),
                          data=SimpleNamespace(tags={'architecture_id': 'old-model-v1'}))
    accuracy_history.print_specifications(SimpleNamespace(download_artifacts=download), run)
    output = capsys.readouterr().out
    assert 'Model type / architecture: old-model-v1' in output
    assert 'Saved specifications unavailable:' in output
