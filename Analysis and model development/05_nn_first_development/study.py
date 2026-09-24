"""Reproducible historical NN study; no pipeline imports. The original three-window protocol is the default.
Run as a script to execute/resume, or import from notebook 07.
"""
from pathlib import Path
from typing import Any, TypedDict
import copy, hashlib, json, platform, time
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy import sparse
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from IPython.display import display

import argparse
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE
CACHE = Path(os.environ.get("NN_STUDY_CACHE", "/tmp/election_nn_architecture_study_cache"))
DATA_PATH = ROOT / "TEST_TRAIN/train.csv"
EXTRA_DATA_PATH = None
CLASS_LABELS = None
RATES = (0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
ARCHITECTURES = {"16": (16,), "32": (32,), "32_16": (32,16), "64_32": (64,32)}
PATIENCES = (10, 20, 50)
SEEDS = tuple(range(101223,101233))
WINDOWS = ((2015,2017,2019),(2005,2010,2015),(1997,2001,2005))
MAX_EPOCHS = 1000
BATCH_SIZE = 64
FEATURE_COLUMNS = [
    "country/region",
    "previous_winning_party_last_election_vote_share",
    "previous_winner",
    "con_polling",
    "lab_polling",
    "lib_polling",
    "incumbent",
    "previous_con_share",
    "previous_lib_share",
    "previous_lab_share",
    "previous_natSW_share",
    "projected_con_share",
    "projected_lib_share",
    "projected_lab_share",
]
CATEGORICAL_COLUMNS = ["country/region", "previous_winner", "incumbent"]
NUMERIC_COLUMNS = [
    column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS
]



def make_preprocessor() -> ColumnTransformer:
    """Create preprocessing to fit on training rows and reuse for prediction."""
    return ColumnTransformer([
        # Fill missing numbers with training medians, then centre and scale them.
        # Keep even entirely missing columns so the feature layout stays stable.
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), NUMERIC_COLUMNS),
        ("categorical", Pipeline([
            # Treat missing categories as a named category. One-hot encoding makes
            # a separate indicator column for each category seen during fitting;
            # unseen categories at prediction time get all-zero indicators.
            ("impute", SimpleImputer(
                strategy="constant", fill_value="__MISSING__", keep_empty_features=True
            )),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_COLUMNS),
    ])


def as_features(matrix: NDArray[Any] | sparse.spmatrix) -> torch.Tensor:
    """Convert sklearn's feature matrix into a dense float32 PyTorch tensor."""
    # One-hot encoding can produce sparse matrices; these linear layers need dense input.
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    array = np.asarray(matrix, dtype=np.float32)
    # Fail early if preprocessing leaves NaN or infinite values in the predictors.
    if not np.isfinite(array).all():
        raise ValueError("Preprocessed predictors contain non-finite values.")
    return torch.from_numpy(array)



def scores(frame, probs, classes):
    lookup = {label: i for i, label in enumerate(classes)}
    # Unseen evaluation winners have zero probability; report their count.
    indices = np.array([lookup.get(label, -1) for label in frame.winner])
    known = indices >= 0
    target = np.zeros_like(probs, dtype=float)
    target[np.flatnonzero(known), indices[known]] = 1
    winning = np.zeros(len(frame))
    winning[known] = probs[np.flatnonzero(known), indices[known]]
    predictions = np.asarray(classes)[probs.argmax(axis=1)]
    correct = predictions == frame.winner.to_numpy()
    changed = frame.previous_winner.notna() & frame.winner.ne(frame.previous_winner)
    return dict(accuracy=float(correct.mean()),
                changed_seat_accuracy=float(correct[changed.to_numpy()].mean()) if changed.any() else np.nan,
                changed_seats=int(changed.sum()),
                log_loss=float(-np.log(np.clip(winning, np.finfo(float).eps, 1)).mean()),
                brier_score=float(((probs-target)**2).sum(axis=1).mean() + (~known).mean()),
                rows=len(frame), unseen_winners=int((~known).sum()))

class Network(nn.Module):
    def __init__(self, input_dim, widths, output_dim):
        super().__init__()
        layers = []
        for width in widths:
            layers += [nn.Linear(input_dim, width), nn.ReLU()]
            input_dim = width
        self.layers = nn.Sequential(*layers, nn.Linear(input_dim, output_dim))
    def forward(self, x):
        return self.layers(x)


def load_windows():
    data = pd.read_csv(DATA_PATH)
    if EXTRA_DATA_PATH is not None:
        data = pd.concat([data, pd.read_csv(EXTRA_DATA_PATH)], ignore_index=True)
    data['election'] = pd.to_numeric(data.election, errors='raise')
    assert data.election.notna().all() and data.winner.notna().all()
    result = {}
    for cutoff, vy, ey in WINDOWS:
        tr = data.loc[data.election <= cutoff].copy()
        va = data.loc[data.election == vy].copy()
        ev = data.loc[data.election == ey].copy()
        assert len(tr) and len(va) and len(ev) and tr.election.max() == cutoff < vy < ey
        encoder = LabelEncoder().fit(tr.winner if CLASS_LABELS is None else CLASS_LABELS)
        assert set(va.winner).issubset(encoder.classes_)
        prep = make_preprocessor()
        xt = as_features(prep.fit_transform(tr[FEATURE_COLUMNS]))
        xv = as_features(prep.transform(va[FEATURE_COLUMNS]))
        xe = as_features(prep.transform(ev[FEATURE_COLUMNS]))
        yt = torch.tensor(encoder.transform(tr.winner), dtype=torch.long)
        yv = torch.tensor(encoder.transform(va.winner), dtype=torch.long)
        result[cutoff] = dict(training=tr, validation=va, evaluation=ev, xt=xt, xv=xv, xe=xe,
                              yt=yt, yv=yv, classes=encoder.classes_, validation_year=vy, evaluation_year=ey)
    return result


def cache_path(cutoff, arch, rate, seed):
    return CACHE / f'{cutoff}_{arch}_lr{rate:g}_seed{seed}.npz'


def fit_run(window, arch, rate, seed):
    """One SGD path implements all patience rules exactly, without triple training.

    At the first stop for each patience, freeze that rule's best checkpoint and
    predictions. Longer-patience rules continue the same deterministic trajectory.
    Predictions never influence weight updates, checkpointing or stopping.
    """
    records, vp, ep, history = [], [], [], []
    active = set(PATIENCES)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = Network(window['xt'].shape[1], ARCHITECTURES[arch], len(window['classes']))
        loader = DataLoader(TensorDataset(window['xt'], window['yt']), batch_size=BATCH_SIZE,
                            shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
        opt = torch.optim.SGD(net.parameters(), lr=rate, momentum=0., weight_decay=0.)
        criterion = nn.CrossEntropyLoss()
        best, best_epoch, best_state, stale = float('inf'), 0, None, 0
        started = time.monotonic()
        for epoch in range(1, MAX_EPOCHS + 1):
            net.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = criterion(net(xb), yb)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite training loss: {arch}, {rate}, {seed}, epoch {epoch}')
                loss.backward(); opt.step()
            net.eval()
            with torch.inference_mode():
                vl = criterion(net(window['xv']), window['yv']).item()
            if not np.isfinite(vl):
                raise RuntimeError('Nonfinite validation loss')
            history.append(vl)
            if vl < best:
                best, best_epoch, stale = vl, epoch, 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                stale += 1
            for patience in sorted(active.copy()):
                if stale >= patience or epoch == MAX_EPOCHS:
                    # Copy avoids disturbing training state or consuming random numbers.
                    checkpoint = copy.deepcopy(net)
                    checkpoint.load_state_dict(best_state)
                    with torch.inference_mode():
                        valp = torch.softmax(checkpoint(window['xv']),dim=1).numpy()
                        evalp = torch.softmax(checkpoint(window['xe']),dim=1).numpy()
                    vp.append(valp); ep.append(evalp)
                    records.append(dict(patience=patience, best_epoch=best_epoch, stopping_epoch=epoch,
                                        best_validation_loss=best, hit_epoch_cap=epoch==MAX_EPOCHS,
                                        parameters=sum(p.numel() for p in net.parameters()),
                                        elapsed_seconds=time.monotonic()-started))
                    active.remove(patience)
            if not active:
                break
    order = np.argsort([r['patience'] for r in records])
    return dict(records=np.array(json.dumps([records[i] for i in order])),
                validation=np.array(vp)[order], evaluation=np.array(ep)[order],
                validation_loss_history=np.array(history))


def ensure_runs(windows, architectures, seeds):
    total = len(windows)*len(architectures)*len(RATES)*len(seeds)
    done = 0
    for cutoff, window in windows.items():
        for arch in architectures:
            for rate in RATES:
                for seed in seeds:
                    path = cache_path(cutoff, arch, rate, seed)
                    if not path.exists():
                        fitted = fit_run(window, arch, rate, seed)
                        temporary = path.with_suffix('.tmp.npz')
                        np.savez_compressed(temporary, **fitted)
                        temporary.replace(path)
                    done += 1
                    print(f'{done}/{total} through {cutoff} architecture {arch} LR {rate:g} seed {seed}',flush=True)


def aggregate(windows, architectures, seeds, stage):
    rows, runs, predictions = [], [], []
    for cutoff, window in windows.items():
        for arch in architectures:
            for rate in RATES:
                values = []
                for seed in seeds:
                    with np.load(cache_path(cutoff, arch, rate, seed), allow_pickle=False) as z:
                        values.append((z['validation'].copy(),z['evaluation'].copy()))
                        for pi, record in enumerate(json.loads(str(z['records']))):
                            record.update(stage=stage, train_through=cutoff, validation_year=window['validation_year'],
                                          architecture=arch, learning_rate=rate, seed=seed)
                            record.update({'validation_'+k:v for k,v in scores(window['validation'],z['validation'][pi],window['classes']).items()})
                            runs.append(record)
                for pi, patience in enumerate(PATIENCES):
                    row=dict(stage=stage,train_through=cutoff, validation_year=window['validation_year'],
                             evaluation_year=window['evaluation_year'], architecture=arch, learning_rate=rate,
                             patience=patience, seeds=len(seeds))
                    for split, index in [('validation',0),('evaluation',1)]:
                        probs=np.mean([v[index][pi] for v in values],axis=0)
                        row.update({split+'_'+k:v for k,v in scores(window[split],probs,window['classes']).items()})
                        # Long prediction table preserves probabilities and row alignment for audit.
                        frame=pd.DataFrame(probs,columns=['p_'+str(c) for c in window['classes']])
                        for key, value in dict(stage=stage,split=split,train_through=cutoff,architecture=arch,
                                               learning_rate=rate,patience=patience).items(): frame[key]=value
                        frame['source_row']=window[split].index.to_numpy()
                        frame['winner']=window[split].winner.to_numpy()
                        predictions.append(frame)
                    rows.append(row)
    table=pd.DataFrame(rows)
    table.to_csv(OUTPUT / f'{stage}_results.csv',index=False)
    pd.DataFrame(runs).to_csv(OUTPUT / f'{stage}_training_records.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(OUTPUT / f'{stage}_predictions.csv.gz',index=False,compression='gzip')
    return table


def policy_results(table):
    """Local validation selection only; fixed policies are descriptive comparators."""
    rows=[]
    for (arch,patience,cutoff), group in table.groupby(['architecture','patience','train_through'],sort=False):
        policies={'validation_all_rates':group,
                  'validation_midrange':group.loc[group.learning_rate.isin([.1,.2,.3,.5])],
                  'validation_compact':group.loc[group.learning_rate.isin([.1,.3])],
                  **{f'fixed_{rate:g}':group.loc[group.learning_rate==rate] for rate in RATES}}
        for policy,candidates in policies.items():
            selected=candidates.loc[candidates.validation_log_loss.idxmin()].to_dict()
            selected.update(policy=policy)
            rows.append(selected)
    # Also compare choosing both patience and LR using this window's validation loss.
    for (arch,cutoff),group in table.groupby(['architecture','train_through'],sort=False):
        selected=group.loc[group.validation_log_loss.idxmin()].to_dict()
        selected.update(policy='validation_rate_and_patience')
        selected['patience']='selected'
        rows.append(selected)
    return pd.DataFrame(rows)


def summarize(policies):
    return policies.groupby(['architecture','patience','policy'],sort=False).agg(
        mean_accuracy=('evaluation_accuracy','mean'),worst_accuracy=('evaluation_accuracy','min'),
        mean_changed_accuracy=('evaluation_changed_seat_accuracy','mean'),
        mean_log_loss=('evaluation_log_loss','mean'),mean_brier=('evaluation_brier_score','mean'),
        mean_validation_log_loss=('validation_log_loss','mean'),
    ).reset_index().sort_values(['mean_accuracy','mean_log_loss'],ascending=[False,True],kind='stable')


def baselines(windows):
    rows=[]
    for cutoff,w in windows.items():
        ev=w['evaluation']; classes=w['classes']
        for name in ['previous_winner','training_majority']:
            pred=ev.previous_winner.to_numpy() if name=='previous_winner' else np.repeat(w['training'].winner.mode().iloc[0],len(ev))
            changed=ev.previous_winner.notna() & ev.winner.ne(ev.previous_winner)
            correct=pred==ev.winner.to_numpy()
            rows.append(dict(train_through=cutoff,evaluation_year=w['evaluation_year'],baseline=name,
                             accuracy=correct.mean(),changed_seat_accuracy=correct[changed].mean(),rows=len(ev)))
    pd.DataFrame(rows).to_csv(OUTPUT/'baselines.csv',index=False)


def main():
    torch.set_num_threads(1)
    CACHE.mkdir(parents=True,exist_ok=True)
    configuration=dict(data_sha256=hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
                       implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       architectures=ARCHITECTURES,rates=RATES,patiences=PATIENCES,seeds=SEEDS,
                       windows=WINDOWS,max_epochs=MAX_EPOCHS,batch_size=BATCH_SIZE,features=FEATURE_COLUMNS,
                       optimizer='SGD',loss='cross_entropy',min_delta=0.,momentum=0.,weight_decay=0.,
                       torch=torch.__version__,sklearn=sklearn.__version__,numpy=np.__version__,
                       python=platform.python_version(),
                       finalist_rule='Always 32_16 plus challenger with lowest mean locally minimized validation ensemble log loss at patience 20 in 3-seed screen; ties prefer listed architecture order.',
                       interpretation='Development results. Shared elections and retrospective selection preclude independent-test claims.')
    fingerprint=hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest()
    global_cache=CACHE
    # Separate caches by exact data/code/config identity, preventing stale reuse.
    globals()['CACHE']=global_cache/fingerprint[:16]
    CACHE.mkdir(parents=True,exist_ok=True)
    configuration['cache_fingerprint']=fingerprint
    (OUTPUT/'configuration.json').write_text(json.dumps(configuration,indent=2))
    windows=load_windows()
    baselines(windows)
    ensure_runs(windows,list(ARCHITECTURES),SEEDS[:3])
    screen=aggregate(windows,list(ARCHITECTURES),SEEDS[:3],'screening')
    policies=policy_results(screen)
    policies.to_csv(OUTPUT/'screening_policies.csv',index=False)
    summarize(policies).to_csv(OUTPUT/'screening_policy_summary.csv',index=False)
    candidates=policies.loc[(policies.policy=='validation_all_rates') & (policies.patience==20)]
    ranking=candidates.groupby('architecture',sort=False).validation_log_loss.mean().sort_values(kind='stable')
    challenger=next(a for a in ranking.index if a!='32_16')
    finalists=['32_16',challenger]
    (OUTPUT/'finalists.json').write_text(json.dumps(dict(architectures=finalists,validation_ranking=ranking.to_dict()),indent=2))
    print('TEN-SEED FINALISTS '+str(finalists),flush=True)
    ensure_runs(windows,finalists,SEEDS)
    final=aggregate(windows,finalists,SEEDS,'confirmation')
    policies=policy_results(final)
    policies.to_csv(OUTPUT/'confirmation_policies.csv',index=False)
    summarize(policies).to_csv(OUTPUT/'confirmation_policy_summary.csv',index=False)
    print('COMPLETE',flush=True)

if __name__=='__main__':
    main()
