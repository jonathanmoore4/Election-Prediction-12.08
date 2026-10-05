"""Check pipeline wiring and notebook syntax without running data preparation."""
import ast
from importlib import import_module
import json
from pathlib import Path
from unittest.mock import Mock
from contextlib import nullcontext

import pandas as pd
import pytest


def test_pipeline_reuses_prepared_data_and_evaluations_without_preparation_or_training(tmp_path,monkeypatch):
    pipeline=import_module('election.pipeline.run_pipeline')
    prepared={'data_id':'id','train_path':'train.csv','test_path':'test.csv'}
    comparison={'winner_model_id':'logistic_regression','source_run_ids':{'logistic_regression':'run'},'missing_results':[]}
    prepare=Mock(); evaluate=Mock(); final=Mock(return_value={'status':'complete'})
    monkeypatch.setattr(pipeline,'run_preparation',prepare)
    monkeypatch.setattr(pipeline,'evaluate_models',evaluate)
    monkeypatch.setattr(pipeline,'compare_recorded',Mock(return_value=comparison))
    monkeypatch.setattr(pipeline,'evaluate_final',final)
    result=pipeline.run_pipeline(output_dir=tmp_path,prepared_data=prepared,reuse_evaluations=True,
                                 model_ids=['logistic_regression'])
    assert result['status']=='complete'
    prepare.assert_not_called(); evaluate.assert_not_called()
    assert final.call_args.kwargs['selection_run_id']=='run'


def test_complete_pipeline_calls_the_same_independent_stages(tmp_path,monkeypatch):
    pipeline=import_module('election.pipeline.run_pipeline')
    prepared={'data_id':'id'}
    prepare=Mock(return_value=prepared); evaluate=Mock(return_value={'logistic_regression':{}})
    compare=Mock(return_value={'winner_model_id':'logistic_regression','source_run_ids':{'logistic_regression':'run'},'missing_results':[]})
    final=Mock(return_value={'status':'complete'})
    for name,call in [('run_preparation',prepare),('evaluate_models',evaluate),('compare_recorded',compare),('evaluate_final',final)]:
        monkeypatch.setattr(pipeline,name,call)
    result=pipeline.run_pipeline(output_dir=tmp_path,model_ids=['logistic_regression'])
    assert result['prepared_data'] is prepared
    assert prepare.call_count==evaluate.call_count==compare.call_count==final.call_count==1


def test_preparation_preserves_steps_and_stores_only_metadata(tmp_path,monkeypatch):
    preparation=import_module('election.pipeline.preparation')
    raw,cleaned=object(),object()
    frames={'train':pd.DataFrame({'election':[2019],'winner':['con']}),
            'test':pd.DataFrame({'election':[2024],'winner':['lab']})}
    frames['train'].attrs['database_schema']='run_example'
    read=Mock(return_value=raw); clean=Mock(return_value=cleaned); sql=Mock(return_value=frames)
    monkeypatch.setattr(preparation.read_in_raw,'read_raw_data',read)
    monkeypatch.setattr(preparation.clean_data_all,'clean_all_data',clean)
    monkeypatch.setattr(preparation.apply_sql_queries,'apply_sql_queries',sql)
    monkeypatch.setattr(preparation.predictor_guide,'write_predictor_guide',lambda frames,path:path.write_text('guide'))
    monkeypatch.setattr(preparation.apply_sql_queries,'read_snapshot',
                        lambda schema, **kwargs: frames)
    a=preparation.run_preparation({'output_dir':tmp_path})
    clean.assert_called_once_with(raw); sql.assert_called_once_with(cleaned)
    original_manifest=next(tmp_path.glob('datasets/*/prepared_data.json'))
    old_content=original_manifest.read_text()
    assert not list(tmp_path.rglob('*.csv'))
    assert 'train_path' not in a and 'test_path' not in a
    frames['train'].loc[0,'winner']='lab'
    b=preparation.run_preparation({'output_dir':tmp_path})
    assert a['data_id']!=b['data_id']
    assert original_manifest.read_text()==old_content
    assert json.loads((tmp_path/'latest_prepared.json').read_text())==b
    assert a['database_schema']=='run_example'


def test_changed_prepared_data_fails_before_evaluation(tmp_path):
    from election.pipeline.preparation import load_prepared
    path=tmp_path/'train.csv'; pd.DataFrame({'election':[2019],'winner':['con']}).to_csv(path,index=False)
    with pytest.raises(ValueError,match='changed'):
        load_prepared({'data_id':'wrong','train_path':str(path)})


def test_comparison_does_not_import_models_or_load_data(monkeypatch):
    pipeline=import_module('election.pipeline.run_pipeline')
    monkeypatch.setattr(pipeline,'model_functions',Mock(side_effect=AssertionError('imported models')))
    monkeypatch.setattr(pipeline,'load_prepared',Mock(side_effect=AssertionError('loaded data')))
    compare=Mock(return_value={'winner_model_id':None})
    monkeypatch.setattr(pipeline,'compare_models',compare)
    pipeline.compare_recorded({'data_id':'id'},model_ids=['logistic_regression'])
    assert compare.call_count==1


def test_missing_models_prevent_automatic_final_evaluation(tmp_path,monkeypatch):
    pipeline=import_module('election.pipeline.run_pipeline')
    monkeypatch.setattr(pipeline,'compare_recorded',Mock(return_value={'missing_results':[{'model_id':'nn01'}]}))
    final=Mock(); monkeypatch.setattr(pipeline,'evaluate_final',final)
    with pytest.raises(ValueError,match='requires all requested'):
        pipeline.run_pipeline(output_dir=tmp_path,prepared_data={'data_id':'id'},reuse_evaluations=True)
    final.assert_not_called()


def test_pipeline_notebook_code_compiles_and_uses_root_output_directory():
    path = Path(__file__).resolve().parents[1]/'notebooks/00_run_pipeline.ipynb'
    notebook = json.loads(path.read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    assert 'output_directory = project_root / "outputs"' in ''.join(
        ''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code'
    )


def test_pipeline_notebook_imports_callable_entry_point():
    path = Path(__file__).resolve().parents[1]/'notebooks/00_run_pipeline.ipynb'
    notebook = json.loads(path.read_text())
    setup = next(cell for cell in notebook['cells']
                 if cell['cell_type'] == 'code')
    namespace = {}
    exec(''.join(setup['source']), namespace)
    assert callable(namespace['run_pipeline'])
    assert namespace['run_pipeline'] is import_module(
        'election.pipeline.run_pipeline').run_pipeline


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


def test_final_uses_recorded_grid_and_fixed_settings_and_checks_compatibility(tmp_path,monkeypatch):
    from election.models.config import MODEL_COMPARISON_SCHEDULE
    from election.models.evaluation import json_value
    pipeline=import_module('election.pipeline.run_pipeline')
    train=pd.DataFrame({'election':[2019],'winner':['con']})
    test=pd.DataFrame({'election':[2024],'winner':['lab']})
    train_path=tmp_path/'train.csv'; test_path=tmp_path/'test.csv'
    train.to_csv(train_path,index=False); test.to_csv(test_path,index=False)
    prepared={'train_path':str(train_path),'test_path':str(test_path)}
    _,locations=pipeline.load_prepared(prepared)
    function,candidates,metadata=pipeline.model_functions('logistic_regression')
    selected={'model_id':'logistic_regression','schedule':json_value(MODEL_COMPARISON_SCHEDULE),
        'metadata':{**metadata,**pipeline._data_metadata(locations),'fixed_settings':{'max_iter':23}},
        'hyperparameter_candidates':{'C':[.123]}}
    monkeypatch.setattr(pipeline,'read_evaluation',Mock(return_value=selected))
    nested=Mock(return_value={'status':'complete'}); monkeypatch.setattr(pipeline,'nested_cv',nested)
    pipeline.evaluate_final(prepared,selection_run_id='historical')
    assert nested.call_args.args[1]=={'C':[.123]}
    assert nested.call_args.kwargs['metadata']['fixed_settings']=={'max_iter':23}
    assert nested.call_args.kwargs['metadata']['selection_run_id']=='historical'
    selected['metadata']['architecture_id']='incompatible'
    with pytest.raises(ValueError,match='architecture_id'):
        pipeline.evaluate_final(prepared,selection_run_id='historical')
    selected['metadata']['architecture_id']=metadata['architecture_id']
    selected['metadata']['features']=['different_feature']
    nested.reset_mock()
    with pytest.raises(ValueError,match='features'):
        pipeline.evaluate_final(prepared,selection_run_id='historical')
    nested.assert_not_called()


def test_database_snapshot_is_preferred_to_csv_and_detects_changes(monkeypatch):
    preparation=import_module('election.pipeline.preparation')
    frames={'train':pd.DataFrame({'election':[2019],'winner':['con']}),
            'test':pd.DataFrame({'election':[2024],'winner':['lab']})}
    read=Mock(return_value=frames)
    monkeypatch.setattr(preparation.apply_sql_queries,'read_snapshot',read)
    from election.models.evaluation import data_identity
    prepared={'database_schema':'run_example',
              'data_id':data_identity(frames['train']),
              'test_data_id':data_identity(frames['test']),
              'train_path':'nonexistent.csv','test_path':'nonexistent.csv'}
    history,_=preparation.load_prepared(prepared)
    assert history.election.tolist()==[2019]
    read.assert_called_once_with('run_example',include_test=False)
    combined,_=preparation.load_prepared(prepared,include_test=True)
    assert combined.election.tolist()==[2019,2024]
    read.assert_called_with('run_example',include_test=True)
    frames['test'].loc[0,'winner']='con'
    with pytest.raises(ValueError,match='final test data has changed'):
        preparation.load_prepared(prepared,include_test=True)
    frames['train'].loc[0,'winner']='lab'
    with pytest.raises(ValueError,match='training data has changed'):
        preparation.load_prepared(prepared)


def test_pipeline_accepts_existing_csv_paths_without_a_manifest(tmp_path,monkeypatch):
    pipeline=import_module('election.pipeline.run_pipeline')
    train_path=tmp_path/'train.csv'
    pd.DataFrame({'election':[2019],'winner':['con']}).to_csv(train_path,index=False)
    evaluate=Mock(return_value={})
    monkeypatch.setattr(pipeline,'evaluate_models',evaluate)
    monkeypatch.setattr(pipeline,'compare_recorded',Mock(return_value={'missing_results':[]}))
    result=pipeline.run_pipeline(prepared_data={'train_path':str(train_path)},
        model_ids=['logistic_regression'],final_evaluation=False)
    assert result['prepared_data']['data_id']
    assert evaluate.call_args.args[0]['data_id']==result['prepared_data']['data_id']
