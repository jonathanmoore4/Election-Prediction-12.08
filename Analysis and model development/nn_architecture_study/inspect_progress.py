"""Inspect completed ten-seed, patience-20 ensembles without interrupting a run."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
OUT=HERE/'all_elections'
config=json.loads((OUT/'configuration.json').read_text())
cache=Path(os.environ.get('NN_STUDY_CACHE',str(HERE.parents[1]/'.nn_study_cache')))/('all_elections_'+config['cache_fingerprint'][:16])
data=pd.concat([pd.read_csv(HERE.parents[1]/'TEST_TRAIN/train.csv'),pd.read_csv(HERE.parents[1]/'TEST_TRAIN/test.csv')],ignore_index=True)
rows=[]
classes=np.array(config['classes'])
for cutoff,vy,ey in config['windows']:
    va=data.loc[data.election==vy];ev=data.loc[data.election==ey]
    for arch in config['architectures']:
        candidates=[]
        for rate in config['rates']:
            paths=[cache/f'{cutoff}_{arch}_lr{rate:g}_seed{seed}.npz' for seed in config['seeds']]
            if not all(p.exists() for p in paths):continue
            vp=[];ep=[]
            for p in paths:
                with np.load(p) as z:vp.append(z['validation'][1]);ep.append(z['evaluation'][1])
            vp=np.mean(vp,axis=0);ep=np.mean(ep,axis=0)
            vl=-np.log(np.clip(vp[np.arange(len(va)),np.searchsorted(classes,va.winner)],np.finfo(float).eps,1)).mean()
            pred=classes[ep.argmax(axis=1)];correct=pred==ev.winner.to_numpy();changed=(ev.previous_winner.notna() & ev.winner.ne(ev.previous_winner)).to_numpy()
            candidates.append(dict(election=ey,architecture=arch,rate=rate,validation_loss=float(vl),accuracy=correct.mean(),changed_accuracy=correct[changed].mean(),**{p:int((pred==p).sum()) for p in classes}))
        if len(candidates)==len(config['rates']):
            best=min(candidates,key=lambda r:r['validation_loss']); rows.append(dict(policy='selected',**best))
            rows.append(dict(policy='fixed_0.3',**next(r for r in candidates if r['rate']==.3)))
print(pd.DataFrame(rows).to_string(index=False))
print(f'{len(list(cache.glob("*.npz")))} completed trajectories')
