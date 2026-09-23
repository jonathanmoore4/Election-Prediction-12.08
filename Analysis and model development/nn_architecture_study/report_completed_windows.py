"""Publish a bounded snapshot using only fully completed election windows."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mistune

HERE = Path(__file__).resolve().parent
OUT = HERE / 'all_elections'
ROOT = HERE.parents[1]


def table(frame):
    frame = frame.copy()
    for c in frame:
        if pd.api.types.is_float_dtype(frame[c]):
            frame[c] = frame[c].map(lambda v: f'{v:.2f}')
    return '| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join(['---']*len(frame.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(map(str,r)) + ' |' for r in frame.itertuples(index=False,name=None))


def main():
    config = json.loads((OUT/'configuration.json').read_text())
    cache = Path(os.environ.get('NN_STUDY_CACHE',str(ROOT/'.nn_study_cache'))) / ('all_elections_'+config['cache_fingerprint'][:16])
    data = pd.concat([pd.read_csv(ROOT/'TEST_TRAIN/train.csv'),pd.read_csv(ROOT/'TEST_TRAIN/test.csv')],ignore_index=True)
    classes = np.array(config['classes'])
    complete = []
    rows, parties = [], []
    for cutoff, vy, ey in config['windows']:
        paths = [cache/f'{cutoff}_{a}_lr{r:g}_seed{s}.npz' for a in config['architectures'] for r in config['rates'] for s in config['seeds']]
        if not all(p.exists() for p in paths):
            continue
        complete.append(ey)
        va = data.loc[data.election==vy]; ev = data.loc[data.election==ey]
        known = ev.previous_winner.notna().to_numpy()
        changed = known & ev.winner.ne(ev.previous_winner).to_numpy()
        for arch in config['architectures']:
            candidates = []
            for rate in config['rates']:
                vp, ep = [], []
                for seed in config['seeds']:
                    with np.load(cache/f'{cutoff}_{arch}_lr{rate:g}_seed{seed}.npz',allow_pickle=False) as z:
                        vp.append(z['validation'][1]); ep.append(z['evaluation'][1])
                vp = np.mean(vp,axis=0); ep = np.mean(ep,axis=0)
                np.testing.assert_allclose(ep.sum(1),1,atol=2e-6)
                assert np.isfinite(ep).all() and (ep>=0).all()
                vl = -np.log(np.clip(vp[np.arange(len(va)),np.searchsorted(classes,va.winner)],np.finfo(float).eps,1)).mean()
                candidates.append((float(vl), rate, ep))
            for policy, choice in [('selected',min(candidates,key=lambda r:r[0])),('fixed_0.3',next(c for c in candidates if c[1]==.3))]:
                vl, rate, ep = choice
                pred = classes[ep.argmax(1)]; correct = pred==ev.winner.to_numpy()
                false = known & ~changed & (pred!=ev.previous_winner.to_numpy())
                ll = -np.log(np.clip(ep[np.arange(len(ev)),np.searchsorted(classes,ev.winner)],np.finfo(float).eps,1)).mean()
                baseline = (ev.loc[known,'winner']==ev.loc[known,'previous_winner']).mean()
                assert int(correct[known].sum()) - int((ev.loc[known,'winner']==ev.loc[known,'previous_winner']).sum()) == int((correct&changed).sum())-int(false.sum())
                rows.append(dict(election=ey,architecture=arch,policy=policy,rate=rate,accuracy=100*correct.mean(),changed_accuracy=100*correct[changed].mean(),
                    known_accuracy=100*correct[known].mean(),baseline_known=100*baseline,known_gain_pp=100*(correct[known].mean()-baseline),
                    correct_changes=int((correct&changed).sum()),false_changes=int(false.sum()),missing_history=int((~known).sum()),log_loss=ll,
                    overall_changed_score=50*(correct.mean()+correct[changed].mean())))
                for party in classes:
                    parties.append(dict(election=ey,architecture=arch,policy=policy,party=party,actual=int(ev.winner.eq(party).sum()),predicted=int((pred==party).sum())))
    assert complete, 'No fully completed election windows yet.'
    detail = pd.DataFrame(rows); seats = pd.DataFrame(parties)
    detail.to_csv(OUT/'completed_window_metrics.csv',index=False)
    seats.to_csv(OUT/'completed_window_party_counts.csv',index=False)
    selected = detail.loc[detail.policy=='selected']
    fig, ax = plt.subplots(figsize=(9,4))
    for arch,g in selected.groupby('architecture',sort=False):
        ax.plot(g.election,g.known_accuracy,marker='o',label=arch.replace('_','/'))
    b = selected.drop_duplicates('election')
    ax.plot(b.election,b.baseline_known,'k--',label='Previous winner')
    ax.set(xticks=complete,ylabel='Accuracy on known-history seats (%)',xlabel='Evaluation election',title='Completed windows: validation-selected rates, ten seeds, patience 20')
    ax.grid(alpha=.2); ax.legend(ncol=3); fig.tight_layout()
    chart=OUT/'completed_windows.png';fig.savefig(chart,dpi=160);plt.close(fig)
    count = len(list(cache.glob('*.npz')))
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    sections = [f'# Neural-network architecture across historical elections\n\n**Progress report · {now}**\n',
        f'**The completed evidence does not identify one architecture that performs best across elections and objectives.** The main pattern is variation between elections: architecture changes are smaller than the changes in performance between historical settings. This snapshot covers **{", ".join(map(str,complete))}**, using all four architectures and ten seeds. The full extension to 2024 remains in progress ({count:,} of 2,560 trajectories saved at publication).\n',
        '## Findings to present\n',
        '- **1997 is poorly captured as a change election.** Selected architectures identify only 5.0–6.25% of changed seats correctly. On the 550 constituencies with known previous winners, accuracy is 70.91–72.18%, against 70.91% for predicting the previous winner. A further 91 missing-history seats materially reduce all-seat accuracy.\n'
        '- **High accuracy in 2001 largely reflects stable seats.** The selected models achieve 94.07–94.54% overall accuracy, below the previous-winner baseline of 96.26%. Only 24 seats change winner in the supplied data.\n'
        '- **2005 exposes the cost of false changes.** Selected models identify 61.40–63.16% of changed seats correctly, but overall accuracy is only 78.82–80.57%, against 90.92% for the baseline. The models call too many changes on seats that actually hold.\n'
        '- **2010 favours the larger network, but still shows seat-total bias.** The 64/32 architecture reaches 89.24% accuracy, versus 87.34% for 32/16 and 82.28% for the previous-winner baseline. Its 93 correctly predicted changes exceed its 49 false changes. All four architectures nevertheless overpredict Conservative seats. This election-specific advantage does not establish that the larger architecture is preferable in every period.\n'
        '- **The smaller learning-rate search does not reproduce every full-search choice.** All architectures select rate 1.0 in 1997 and 2001; the original three-window recommendation restricted search to 0.1, 0.2, 0.3 and 0.5. This does not establish that rate 1.0 should be used universally.\n',
        '## Accuracy by election\n\nPercentages below use all supplied evaluation rows. Each architecture selects its rate on the preceding validation election, so differences include both architecture and rate.\n',
        table(selected.pivot(index='election',columns='architecture',values='accuracy').reset_index()),
        '\n## Correctly predicted changes\n\nPercentage of changed seats for which the model predicts the correct new party. Rows with missing previous winners are excluded.\n',
        table(selected.pivot(index='election',columns='architecture',values='changed_accuracy').reset_index()),
        '\n## Average performance on completed windows\n\nEach election receives equal weight. The overall/changed score averages the two accuracy measures, matching the pipeline selector up to a factor of two. Lower log loss indicates better probability predictions. These averages are provisional and exclude incomplete windows.\n',
        table(selected.groupby('architecture',sort=False).agg(mean_accuracy=('accuracy','mean'),mean_changed_accuracy=('changed_accuracy','mean'),mean_overall_changed_score=('overall_changed_score','mean'),mean_log_loss=('log_loss','mean')).reset_index()),
        '\n## Comparison with predicting the previous winner\n\nBoth methods are evaluated on the same known-history rows. A positive gain favours the network. Correct changes minus false changes is the net number of additional correct predictions against this baseline.\n',
        table(selected[['election','architecture','rate','known_accuracy','baseline_known','known_gain_pp','correct_changes','false_changes']]),
        '\n![Accuracy on matching rows](all_elections/completed_windows.png)\n',
        '\n## Fixed learning rate: a cleaner architecture comparison\n\nAll architectures use rate 0.3 and patience 20. Differences still reflect optimisation and model capacity; they do not prove a causal benefit from adding layers.\n',
        table(detail.loc[detail.policy=='fixed_0.3'].pivot(index='election',columns='architecture',values='accuracy').reset_index()),
        '\n## Party seat totals\n\nCounts cover the supplied Great Britain rows, not the entire UK Parliament. `natSW` combines SNP and Plaid Cymru; `oth` pools other parties. These are argmax seat totals, not probabilities of a parliamentary majority.\n']
    for ey in complete:
        g=seats.loc[(seats.policy=='selected')&(seats.election==ey)]
        t=g.pivot(index='party',columns='architecture',values='predicted')
        t.insert(0,'actual',g.drop_duplicates('party').set_index('party').actual)
        sections += [f'\n### {ey}\n',table(t.reset_index())]
    sections += ['\n## Relationship to the completed original study\n',
        'The earlier three-election study separately covered 2005, 2015 and 2019. Its ten-seed 32/16 procedure with full validation-based rate selection and patience 20 achieved 78.82%, 78.80% and 90.19% accuracy respectively. Its mean accuracy was 82.60%, versus 87.22% for the previous-winner baseline. These are prior saved results, not a claim that the current all-architecture extension has finished those later elections. See [the original report](REPORT_three_elections.md).\n',
        '\n## Interpretation and next decision\n',
        'Retain 32/16 as a reference candidate while the full comparison finishes. The current evidence does not support increasing network size as a general solution. Judge a candidate using overall accuracy, correctly predicted changes, false changes, party totals and log loss together. The existing pipeline gives equal weight to overall and changed-seat accuracy; both components are preserved in the accompanying CSV, along with their equally weighted score.\n\n'
        'Before changing the architecture, investigate data representation in a separate controlled experiment: 91 missing-history rows in 1997; national polling features that are constant in the earliest training window; and 115 rows in 2024 whose region names are unseen in training because of naming differences. These identify possible failure mechanisms, not demonstrated causes. No feature corrections have been introduced into this comparison.\n',
        '\n## Method and limits\n',
        'Each window trains through a cutoff, validates on the next whole election and evaluates on the following one. Preprocessing is fitted on training rows only. Architectures are 16, 32, 32/16 and 64/32 ReLU units; SGD rates are 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5 and 1.0. This report uses ten-seed probability ensembles and patience 20, with each seed restoring its lowest validation-loss checkpoint. The full run also evaluates patience 10 and 50.\n\n'
        'This is retrospective development evidence, including previously inspected elections. Ten seeds measure optimisation variability, not ten independent elections. Early evaluation windows have little training history, and later windows reuse earlier elections. The supplied features and boundary mappings have not received a fresh historical-availability audit. This snapshot has checked complete-window coverage, probability validity and the matching-row baseline accounting identity; the full saved-table audit and leave-one-election-out summaries remain pending.\n',
        '\n## Files and resumption\n',
        '- [Completed-window metrics](all_elections/completed_window_metrics.csv)\n- [Party seat counts](all_elections/completed_window_party_counts.csv)\n- [Original three-election report](REPORT_three_elections.md)\n- [All-election notebook](../09_nn_architecture_all_elections.IPYNB) — full result cells remain pending.\n\n'
        'Checkpoints now live in `.nn_study_cache/` inside the workspace. After an interruption, run `python "Analysis and model development/nn_architecture_study/resume_all_elections.py"` using the repository environment. It resumes missing fits, builds the diagnostics and runs the audit. Do not start a duplicate runner while training is active.\n']
    report='\n\n'.join(sections)+'\n'
    (HERE/'REPORT_all_elections_progress.md').write_text(report)
    html=mistune.create_markdown(plugins=['table'])(report)
    uri='data:image/png;base64,'+base64.b64encode(chart.read_bytes()).decode()
    html=html.replace('all_elections/completed_windows.png',uri)
    style='body{font:16px/1.55 system-ui,sans-serif;color:#182638;max-width:1100px;margin:40px auto;padding:0 28px}h1,h2,h3{color:#163e60;line-height:1.2}h1{font-size:32px}h2{margin-top:38px}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}th,td{border-bottom:1px solid #d8e0e8;padding:7px;text-align:right}th{background:#edf3f8}img{max-width:100%}a{color:#17699b}@media print{body{margin:0;font-size:11px}h2,h3{break-after:avoid}tr,img{break-inside:avoid}table{font-size:10px}}'
    (HERE/'REPORT_all_elections_progress.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Neural network election study — progress report</title><style>'+style+'</style><body>'+html+'</body></html>')
    print(f'Published report for {complete}; {count} cached trajectories. Snapshot checks passed.')


if __name__=='__main__':
    main()
