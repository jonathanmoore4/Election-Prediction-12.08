"""Check pipeline wiring and notebook syntax without running data preparation."""
import ast
from importlib import import_module
import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest


@pytest.mark.parametrize("custom_output", [False, True])
def test_pipeline_forwards_registry_and_returns_report_without_real_pipeline_work(tmp_path, monkeypatch, custom_output):
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
    output_dir = tmp_path/'reports' if custom_output else tmp_path/'00_pipeline'/'outputs'
    assert pipeline.run_pipeline(output_dir if custom_output else None, candidates=registry) is result
    select.assert_called_once_with(frames['train'], scores_path=output_dir/'model_accuracies.csv',
                                   candidates=registry, forecast_election=2024)
    pipeline.clean_all_data.assert_called_once_with(raw)
    pipeline.apply_sql_queries.assert_called_once_with(cleaned)
    assert (output_dir/'train.csv').exists()
    assert (output_dir/'test.csv').exists()

    pipeline.write_predictor_guide.assert_called_once_with(
        frames, output_dir/'predictor_descriptions.md')
    assert not (tmp_path/'TEST_TRAIN').exists()

def test_pipeline_notebook_code_compiles_and_uses_root_output_directory():
    path = Path(__file__).resolve().parents[1]/'00_pipeline/00_run_pipeline.IPYNB'
    notebook = json.loads(path.read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    assert 'output_directory = project_root / "00_pipeline" / "outputs"' in ''.join(
        ''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code'
    )


def test_all_notebooks_compile_and_use_current_pipeline_paths():
    root = Path(__file__).resolve().parents[1]
    for path in root.rglob('*'):
        if path.suffix.lower() != '.ipynb' or '.venv' in path.parts:
            continue
        notebook = json.loads(path.read_text())
        for cell in notebook['cells']:
            source = ''.join(cell.get('source', []))
            assert 'TEST_TRAIN' not in source, str(path)
            if cell['cell_type'] == 'code':
                ast.parse(source, filename=str(path))
