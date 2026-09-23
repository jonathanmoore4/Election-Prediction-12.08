"""All-election extension: four architectures, eight rates, three patience rules, ten seeds.

Original three-window artifacts remain available in the parent directory.
Run from any directory; NN_STUDY_WORKERS controls independent fitting processes.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import hashlib
import json
import os
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import torch
import sklearn
import study

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'all_elections'
YEARS = (1987, 1992, 1997, 2001, 2005, 2010, 2015, 2017, 2019, 2024)
WINDOWS = tuple(zip(YEARS[:-2], YEARS[1:-1], YEARS[2:]))
CLASSES = ('con', 'lab', 'lib', 'natSW', 'oth')


def configure():
    study.WINDOWS = WINDOWS
    study.EXTRA_DATA_PATH = study.ROOT / 'TEST_TRAIN/test.csv'
    # Fixed target taxonomy, specified without estimating it from future winners.
    # 1987/1992 have no 'oth' winners; the output still exists for 1997.
    study.CLASS_LABELS = CLASSES
    study.OUTPUT = OUTPUT
    torch.set_num_threads(1)
    return study.load_windows()


def initialize(cache):
    global windows
    windows = configure()
    study.CACHE = Path(cache)


def fit(task):
    cutoff, arch, rate, seed = task
    path = study.cache_path(*task)
    if not path.exists():
        result = study.fit_run(windows[cutoff], arch, rate, seed)
        temporary = path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, **result)
        temporary.replace(path)
    return task


def main():
    OUTPUT.mkdir(exist_ok=True)
    windows = configure()
    configuration = dict(
        data_sha256=hashlib.sha256(study.DATA_PATH.read_bytes()).hexdigest(),
        evaluation_data_sha256=hashlib.sha256(study.EXTRA_DATA_PATH.read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(study.__file__).read_bytes()).hexdigest(),
        extension_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        architectures=study.ARCHITECTURES, rates=study.RATES, patiences=study.PATIENCES,
        seeds=study.SEEDS, windows=WINDOWS, classes=CLASSES,
        max_epochs=study.MAX_EPOCHS, batch_size=study.BATCH_SIZE,
        features=study.FEATURE_COLUMNS, optimizer='SGD', loss='cross_entropy',
        min_delta=0., momentum=0., weight_decay=0.,
        torch=torch.__version__, sklearn=sklearn.__version__, numpy=np.__version__,
        python=platform.python_version(),
        finalist_rule='All four architectures confirmed with ten seeds; no cross-election shortlist.',
        interpretation='Retrospective development, including 2024. Each window selects rates/checkpoints only on its preceding validation election.',
        target_taxonomy='Fixed five classes. No oth training winners in 1987 or through 1992.',
    )
    fingerprint=hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest()
    study.CACHE = study.CACHE / ('all_elections_' + fingerprint[:16])
    study.CACHE.mkdir(parents=True, exist_ok=True)
    configuration['cache_fingerprint'] = fingerprint
    (OUTPUT/'configuration.json').write_text(json.dumps(configuration, indent=2))
    (OUTPUT/'finalists.json').write_text(json.dumps({'architectures':list(study.ARCHITECTURES)},indent=2))
    study.baselines(windows)
    tasks = [(cutoff,arch,rate,seed) for cutoff in windows for arch in study.ARCHITECTURES
             for rate in study.RATES for seed in study.SEEDS]
    pending = [t for t in tasks if not study.cache_path(*t).exists()]
    print(f'{len(tasks)} trajectories; {len(pending)} pending', flush=True)
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context('fork'), max_workers=int(os.environ.get('NN_STUDY_WORKERS','2')),
                             initializer=initialize, initargs=(str(study.CACHE),)) as pool:
        for count, task in enumerate(pool.map(fit, pending, chunksize=1), 1):
            if count % 10 == 0 or count == len(pending):
                print(f'{count}/{len(pending)} completed: {task}', flush=True)
    for stage,seeds in [('screening',study.SEEDS[:3]),('confirmation',study.SEEDS)]:
        table = study.aggregate(windows, list(study.ARCHITECTURES), seeds, stage)
        policies = study.policy_results(table)
        policies.to_csv(OUTPUT/f'{stage}_policies.csv',index=False)
        study.summarize(policies).to_csv(OUTPUT/f'{stage}_policy_summary.csv',index=False)
    print('COMPLETE',flush=True)


if __name__ == '__main__':
    main()
