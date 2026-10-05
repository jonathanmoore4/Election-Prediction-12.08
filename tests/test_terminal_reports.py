"""Saved reports expose pipeline results without running training stages."""
import json

from election.reports import comparison, prepared_data, run_details


def test_comparison_reports_partial_ranking_and_selection(monkeypatch, capsys):
    calls = []

    def compare(data, *, model_ids, tracking):
        calls.append((str(data), model_ids, tracking))
        return dict(ranked_results=[dict(model_id='logistic_regression',
                    mean_outer_accuracy=.8125, run_id='historical-run')],
                    winner_model_id='logistic_regression',
                    source_run_ids={'logistic_regression': 'historical-run'},
                    missing_results=[dict(model_id='random_forest', reason='No saved result.')],
                    excluded_results=[])

    monkeypatch.setattr(comparison, 'compare_recorded', compare)
    comparison.main(['--data-dir', 'saved.json', '--model', 'logistic_regression',
                     'random_forest', '--tracking-uri', 'http://localhost:5000'])
    output = capsys.readouterr().out
    assert '81.25%' in output
    assert 'Selection run ID: historical-run' in output
    assert 'Comparison incomplete' in output
    assert 'Missing random_forest: No saved result.' in output
    assert calls[0][:2] == ('saved.json', ['logistic_regression', 'random_forest'])


def test_prepared_report_reads_manifest_and_resolves_guide(tmp_path, capsys):
    manifest = tmp_path / 'prepared_data.json'
    manifest.write_text(json.dumps(dict(data_id='dataset-id', database_schema='retained',
                                        predictor_guide_path='guide.md')))
    prepared_data.main(['--data-dir', str(tmp_path)])
    output = capsys.readouterr().out
    assert 'Data id: dataset-id' in output
    assert 'Database schema: retained' in output
    assert str(tmp_path / 'guide.md') in output


def test_run_details_reports_historical_tuning_and_scores(monkeypatch, capsys):
    def read(run_id, *, tracking):
        assert run_id == 'historical-run'
        return dict(run_id=run_id, model_id='logistic_regression',
                    schedule={'purpose': 'model_comparison'}, mean_outer_accuracy=.8,
                    metadata=dict(architecture_id='logistic_regression-v1',
                                  training_protocol='full-history-v1', fixed_settings={}),
                    hyperparameter_candidates={'C': [.1, 1]},
                    outer_results={'2019': dict(accuracy=.8, hyperparameters={'C': .1},
                        mean_inner_accuracy=.75, inner_accuracies={'2017': .75},
                        training_years=[2015, 2017], refit_durations={})})

    monkeypatch.setattr(run_details, 'read_evaluation', read)
    run_details.main(['historical-run'])
    output = capsys.readouterr().out
    assert 'Election 2019: 80.00% accuracy' in output
    assert 'Selected hyperparameters: {"C": 0.1}' in output
    assert 'Inner election accuracies: {"2017": "75.00%"}' in output
