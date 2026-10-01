"""Check pipeline wiring and notebook syntax without running data preparation."""
import ast
from importlib import import_module
import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd


def test_pipeline_forwards_registry_and_returns_report_without_real_pipeline_work(tmp_path, monkeypatch):
    pipeline = import_module('Run Pipeline.run_pipeline')
    monkeypatch.setattr(pipeline, 'PROJECT_ROOT', tmp_path)
    raw, cleaned = object(), object()
    frames = {'train': pd.DataFrame({'election':[2019]}),
              'test': pd.DataFrame({'election':[2024]})}
    monkeypatch.setattr(pipeline, 'read_raw_data', Mock(return_value=raw))
    monkeypatch.setattr(pipeline, 'clean_all_data', Mock(return_value=cleaned))
    monkeypatch.setattr(pipeline, 'apply_sql_queries', Mock(return_value=frames))
    monkeypatch.setattr(pipeline, 'write_predictor_guide', Mock())
    result = object()
    select = Mock(return_value=result)
    monkeypatch.setattr(pipeline, 'automated_model_selection', select)
    registry = [object()]
    assert pipeline.run_pipeline(tmp_path/'reports', candidates=registry) is result
    select.assert_called_once_with(frames['train'], scores_path=tmp_path/'reports'/'model_accuracies.csv',
                                   candidates=registry, forecast_election=2024)
    pipeline.clean_all_data.assert_called_once_with(raw)
    pipeline.apply_sql_queries.assert_called_once_with(cleaned)
    assert (tmp_path/'TEST_TRAIN'/'train.csv').exists()
    assert (tmp_path/'TEST_TRAIN'/'test.csv').exists()


def test_pipeline_notebook_code_compiles_and_contains_no_stale_outputs():
    path = Path(__file__).resolve().parents[1]/'Analysis and model development/00_pipeline/00_run_pipeline.IPYNB'
    notebook = json.loads(path.read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
            assert cell['execution_count'] is None
            assert cell['outputs'] == []
