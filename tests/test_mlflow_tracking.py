"""Compact MLflow records, completion failures and newest-compatible selection."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from election.models import tracking
from election.models.compare_models import compare_models
from tests.test_automated_model_selection import evaluated


class MemoryClient:
    """Native MLflow call surface; deterministic failures without a server."""
    def __init__(self, directory):
        self.directory=directory; self.runs={}; self.clock=0; self.fail=None
    def create_run(self, experiment_id, tags):
        run_id=f'{len(self.runs)+1:032x}'
        run=SimpleNamespace(info=SimpleNamespace(run_id=run_id,status='RUNNING',end_time=None),
                            data=SimpleNamespace(tags=dict(tags),metrics={},params={}))
        self.runs[run_id]=run
        return run
    def log_dict(self, run_id, value, filename):
        if self.fail == 'artifact': raise RuntimeError('artifact unavailable')
        directory=self.directory/run_id; directory.mkdir()
        (directory/filename).write_text(json.dumps(value,allow_nan=False))
    def log_metric(self, run_id, key, value):
        if self.fail == 'metric': raise RuntimeError('metric unavailable')
        self.runs[run_id].data.metrics[key]=value
    def log_param(self,run_id,key,value): self.runs[run_id].data.params[key]=str(value)
    def set_tag(self,run_id,key,value): self.runs[run_id].data.tags[key]=value
    def set_terminated(self,run_id,status):
        if self.fail == 'finish' and status == 'FINISHED': raise RuntimeError('finish unavailable')
        self.clock+=1; self.runs[run_id].info.status=status; self.runs[run_id].info.end_time=self.clock
    def get_run(self,run_id): return self.runs[run_id]
    def download_artifacts(self,run_id,filename): return str(self.directory/run_id/filename)
    def search_runs(self,*args,**kwargs):
        class Page(list): token=None
        return Page(sorted(self.runs.values(),key=lambda r:-(r.info.end_time or 0)))


@pytest.fixture
def client(tmp_path,monkeypatch):
    client=MemoryClient(tmp_path)
    monkeypatch.setattr(tracking,'_client',lambda settings:(client,'experiment'))
    return client


def record(result,client):
    run_id=tracking.log_evaluation_result(result,tracking={})
    return client.get_run(run_id)


def compatibility(result): return tracking.compatibility_fields(result['metadata'],result['schedule'])


def test_one_run_six_metrics_one_small_summary_and_no_observations(client):
    result=evaluated(); run=record(result,client)
    assert len(client.runs)==1
    assert run.info.status=='FINISHED'
    assert len(run.data.metrics)==6
    path=client.directory/run.info.run_id/'evaluation.json'
    artifact=json.loads(path.read_text())
    assert path.stat().st_size < 15000
    assert list((client.directory/run.info.run_id).iterdir())==[path]
    assert artifact['metadata']['hyperparameter_search'] == {
        'sampler': 'optuna-random', 'n_trials': 30, 'random_seed': 42}
    assert artifact['outer_results']['2019']['accuracy']==0
    assert 'fit_records' not in path.read_text()
    assert 'constituency_id' not in path.read_text()
    assert 'mlflow.parentRunId' not in run.data.tags
    assert tracking.read_evaluation(run.info.run_id,tracking={})['mean_outer_accuracy']==.8


@pytest.mark.parametrize('failure',['artifact','metric','finish'])
def test_logging_failure_marks_run_failed_and_ineligible(client,failure):
    result=evaluated(); client.fail=failure
    with pytest.raises(RuntimeError,match='unavailable'): record(result,client)
    run=next(iter(client.runs.values()))
    assert run.info.status=='FAILED'
    assert run.data.tags['record_complete']=='false'
    comparison=compare_models(model_ids=['constant'],compatibility=compatibility(result),tracking={})
    assert comparison['winner_model_id'] is None
    assert len(comparison['missing_results'])==1


def test_partial_evaluation_cannot_create_a_completed_record(client):
    result=evaluated(); result['outer_results'].pop(2019)
    with pytest.raises(ValueError,match='Incomplete evaluation'): record(result,client)
    assert not client.runs


def test_latest_compatible_run_wins_over_historically_highest_score(client):
    result=evaluated(); old=record(result,client)
    newer=deepcopy(result)
    for fold in newer['outer_results'].values(): fold['accuracy']=.2
    newer['mean_outer_accuracy']=.2
    latest=record(newer,client)
    comparison=compare_models(model_ids=['constant'],compatibility=compatibility(result),tracking={})
    assert comparison['source_run_ids']=={'constant':latest.info.run_id}
    assert comparison['ranked_results'][0]['mean_outer_accuracy']==.2
    assert old.info.run_id != latest.info.run_id


def test_incompatible_and_final_runs_do_not_replace_latest_historical_result(client):
    result=evaluated(); valid=record(result,client)
    other=deepcopy(result); other['metadata']['data_id']='different'; other['data_id']='different'
    record(other,client)
    final=record(result,client); final.data.tags['purpose']='final_evaluation'
    comparison=compare_models(model_ids=['constant','missing'],compatibility=compatibility(result),tracking={})
    assert comparison['source_run_ids']=={'constant':valid.info.run_id}
    assert len(comparison['excluded_results'])==2
    assert comparison['missing_results'][0]['model_id']=='missing'


@pytest.mark.parametrize('corruption',['missing_metric','nonfinite','artifact_mean','artifact_run_id','artifact_schedule'])
def test_invalid_latest_summary_is_excluded_and_previous_valid_run_is_used(client,corruption):
    result=evaluated(); previous=record(result,client); run=record(result,client)
    path=client.directory/run.info.run_id/'evaluation.json'
    artifact=json.loads(path.read_text())
    if corruption=='missing_metric': run.data.metrics.pop('accuracy_2019')
    elif corruption=='nonfinite': run.data.metrics['mean_outer_accuracy']=float('nan')
    elif corruption=='artifact_mean': artifact['mean_outer_accuracy']=.99
    elif corruption=='artifact_run_id': artifact['run_id']='another'
    else: artifact['schedule']['name']='another'
    path.write_text(json.dumps(artifact))
    comparison=compare_models(model_ids=['constant'],compatibility=compatibility(result),tracking={})
    assert comparison['source_run_ids']['constant']==previous.info.run_id
    assert comparison['excluded_results']


def test_model_ties_follow_requested_model_order(client):
    first=evaluated('first'); second=evaluated('second')
    record(first,client); record(second,client)
    comparison=compare_models(model_ids=['second','first'],compatibility=compatibility(first),tracking={})
    assert comparison['winner_model_id']=='second'


def test_pagination_does_not_omit_requested_models(client,monkeypatch):
    result=evaluated(); valid=record(result,client)
    other=evaluated('other'); irrelevant=record(other,client)
    class Page(list): pass
    def pages(*args,**kwargs):
        if kwargs['page_token'] is None:
            page=Page([irrelevant]); page.token='next'
        else:
            page=Page([valid]); page.token=None
        return page
    monkeypatch.setattr(client,'search_runs',pages)
    records=tracking.latest_completed_evaluations(model_ids=['constant'],compatibility=compatibility(result),tracking={})
    assert records['evaluations']['constant']['run_id']==valid.info.run_id


def test_real_server_records_and_reads_compact_evaluation(tracking_settings):
    result=evaluated(tracking=tracking_settings)
    client,_=tracking._client(tracking_settings)
    run=client.get_run(result['run_id'])
    assert run.info.status=='FINISHED'
    assert len(run.data.metrics)==6
    assert [a.path for a in client.list_artifacts(result['run_id'])]==['evaluation.json']
    assert tracking.read_evaluation(result['run_id'],tracking=tracking_settings)['mean_outer_accuracy']==.8
    comparison=compare_models(model_ids=['constant'],compatibility=compatibility(result),tracking=tracking_settings)
    assert comparison['winner_model_id']=='constant'
