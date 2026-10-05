"""Synthetic integration check of compact MLflow runs and pipeline stages."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from election.models.config import OUTER_ELECTIONS
from election.models.tracking import read_evaluation, _client
from election.pipeline import run_pipeline as pipeline
from election.pipeline.preparation import run_preparation


def demo_model(train, test, parameters, *, fit_records=None, return_details=False,
               cache=None, output_dir=None):
    assert train.election.max() < test.election.min()
    accuracy = float(test.winner.eq(parameters['party']).mean())
    if output_dir is not None:
        import numpy as np
        from election.models.model_function import export_predictions
        export_predictions(test, np.full(len(test), parameters['party']), output_dir, 'demo')
    return dict(accuracy=accuracy, training_years=sorted(train.election.unique()),
                evaluation_rows=len(test), excluded_rows=0) if return_details else accuracy


def test_synthetic_pipeline_records_compact_runs(tracking_settings):
    tracking = tracking_settings
    models = list(pipeline.MODEL_MODULES)
    train = pd.DataFrame([dict(election=year, winner='con', previous_winner='con')
        for year in (1987,1992,1997,2001,*OUTER_ELECTIONS) for _ in range(2)])
    test = train.query('election == 2019').assign(election=2024)
    train.attrs['database_schema'] = 'synthetic_snapshot'
    def functions(model_id):
        return demo_model, {'party':['con','lab']}, dict(model_id=model_id,
            architecture_id='demo-v1', training_protocol='full-history-v1',
            features=['previous_winner'], fixed_settings={})
    with TemporaryDirectory(prefix='election-mlflow-validation-') as temporary:
        output = Path(temporary)
        with patch('election.pipeline.preparation.read_in_raw.read_raw_data', return_value={}), \
             patch('election.pipeline.preparation.clean_data_all.clean_all_data', return_value={}), \
             patch('election.pipeline.preparation.apply_sql_queries.apply_sql_queries', return_value={'train':train,'test':test}), \
             patch('election.pipeline.preparation.predictor_guide.write_predictor_guide', side_effect=lambda frames,path:path.write_text('synthetic data')), \
             patch('election.pipeline.preparation.apply_sql_queries.read_snapshot', return_value={'train':train,'test':test}):
            prepared = run_preparation({'output_dir':output})
        reports = []
        with patch.object(pipeline, 'model_functions', side_effect=functions), \
             patch('election.pipeline.preparation.apply_sql_queries.read_snapshot', return_value={'train':train,'test':test}):
            for _ in range(2):
                result = pipeline.run_pipeline(output_dir=output, prepared_data=prepared,
                                               tracking=tracking, verbose=False)
                assert result['status'] == 'complete'
                assert result['comparison']['winner_model_id'] == models[0]
                assert result['final_evaluation']['mean_outer_accuracy'] == 1
                reports.append(result)
            comparison = pipeline.compare_recorded(prepared, tracking=tracking)
            assert comparison['source_run_ids'] == reports[-1]['comparison']['source_run_ids']
        client, experiment_id = _client(tracking)
        runs = client.search_runs([experiment_id], max_results=100)
        assert len(runs) == 18
        for run in runs:
            assert run.info.status == 'FINISHED'
            assert 'mlflow.parentRunId' not in run.data.tags
            assert len(run.data.metrics) == (6 if run.data.tags['purpose'] == 'model_comparison' else 2)
            assert [a.path for a in client.list_artifacts(run.info.run_id)] == ['evaluation.json']
        original_id = reports[0]['evaluations'][models[0]]['run_id']
        assert read_evaluation(original_id, tracking=tracking)['mean_outer_accuracy'] == 1
        for filename in ('test_predictions.csv','test_confusion_matrix.csv','test_confusion_matrix.png'):
            assert (output/filename).exists()
