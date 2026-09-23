"""Resume the seven-window study using only the retained historical input.

The original per-seed caches are reusable: the included study.py is byte-identical,
and training/validation/evaluation data for these windows are unchanged.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import study
from fast_batches import TensorLoader

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'


def configure():
    config = json.loads((OUT/'configuration.json').read_text())
    study.DATA_PATH = HERE/'data/train.csv'
    assert hashlib.sha256(study.DATA_PATH.read_bytes()).hexdigest() == config['data_sha256']
    assert hashlib.sha256(Path(study.__file__).read_bytes()).hexdigest() == config['implementation_sha256']
    for library, version in [('torch',torch.__version__),('numpy',np.__version__),('sklearn',study.sklearn.__version__)]:
        if version != config[library]:
            raise RuntimeError(f'{library} version differs from recorded experiment; do not mix caches.')
    study.EXTRA_DATA_PATH = None
    study.WINDOWS = tuple(tuple(w) for w in config['windows'])
    assert max(w[2] for w in study.WINDOWS) == 2019
    study.CLASS_LABELS = tuple(config['classes'])
    study.OUTPUT = OUT
    study.CACHE = Path(os.environ.get('NN_STUDY_CACHE',str(HERE/'.nn_study_cache'))) / ('all_elections_'+config['source_cache_fingerprint'][:16])
    study.CACHE.mkdir(parents=True,exist_ok=True)
    study.DataLoader = TensorLoader
    torch.set_num_threads(1)
    return study.load_windows()


def initialize():
    global windows
    windows=configure()


def fit(task):
    cutoff,arch,rate,seed=task
    path=study.cache_path(*task)
    if not path.exists():
        result=study.fit_run(windows[cutoff],arch,rate,seed)
        temporary=path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary,**result)
        temporary.replace(path)
    return task


def main():
    windows=configure()
    tasks=[(cutoff,arch,rate,seed) for cutoff in windows for arch in study.ARCHITECTURES for rate in study.RATES for seed in study.SEEDS]
    pending=[t for t in tasks if not study.cache_path(*t).exists()]
    print(f'{len(tasks)} trajectories through 2019; {len(pending)} pending.',flush=True)
    with ProcessPoolExecutor(max_workers=int(os.environ.get('NN_STUDY_WORKERS','2')),mp_context=multiprocessing.get_context('fork'),initializer=initialize) as pool:
        for i,_ in enumerate(pool.map(fit,pending),1):
            if i%10==0:print(f'{i}/{len(pending)} completed.',flush=True)
    study.baselines(windows)
    for stage,seeds in [('screening',study.SEEDS[:3]),('confirmation',study.SEEDS)]:
        result=study.aggregate(windows,list(study.ARCHITECTURES),seeds,stage)
        policy=study.policy_results(result)
        policy.to_csv(OUT/f'{stage}_policies.csv',index=False)
        study.summarize(policy).to_csv(OUT/f'{stage}_policy_summary.csv',index=False)
    for script in ['summarize_results.py','verify_results.py','build_report.py']:
        subprocess.run([sys.executable,str(HERE/script)],check=True)


if __name__=='__main__':main()
