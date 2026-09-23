"""Rebuild election/party/transition diagnostics, figures and a numerical report."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study_inputs import load_data

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
ROOT = HERE.parents[1]
ARCHS = ['16', '32', '32_16', '64_32']
LABELS = {'16':'16', '32':'32', '32_16':'32/16', '64_32':'64/32'}
PARTIES = ['con','lab','lib','natSW','oth']


def table(frame, percent=(), digits=3):
    frame = frame.copy()
    for c in frame:
        if c in percent:
            frame[c] = frame[c].map(lambda v: f'{100*v:.2f}%')
        elif pd.api.types.is_float_dtype(frame[c]):
            frame[c] = frame[c].map(lambda v: f'{v:.{digits}f}')
    return '| ' + ' | '.join(map(str,frame.columns)) + ' |\n| ' + ' | '.join(['---']*len(frame.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(map(str,row)) + ' |' for row in frame.itertuples(index=False,name=None))


def main():
    config = json.loads((OUT/'configuration.json').read_text())
    data = load_data(OUT)
    policies = pd.read_csv(OUT/'confirmation_policies.csv')
    results = pd.read_csv(OUT/'confirmation_results.csv')
    baseline = pd.read_csv(OUT/'baselines.csv')
    predictions = pd.read_csv(OUT/'confirmation_predictions.csv.gz')
    selected = policies.loc[(policies.patience.astype(str)=='20') & policies.policy.isin(['validation_all_rates','fixed_0.3'])].copy()
    rows, parties, transitions, confusions, paired, coverage = [], [], [], [], [], []
    seed_rows = []
    feature_ranges = []
    history_party = []
    cache = Path(os.environ.get('NN_STUDY_CACHE',str(HERE/'.nn_study_cache'))) / ('all_elections_' + config['source_cache_fingerprint'][:16])
    saved = {}
    pcols = ['p_'+p for p in PARTIES]
    for cutoff,vy,ey in config['windows']:
        tr = data.loc[data.election<=cutoff]
        ev = data.loc[data.election==ey]
        for feature in ['Conservative','Labour','LD']:
            minimum,maximum = tr[feature].min(),tr[feature].max()
            value = ev[feature].iloc[0]
            assert ev[feature].nunique() == 1
            feature_ranges.append(dict(evaluation_year=ey,feature=feature,
                training_min=minimum,training_max=maximum,evaluation_value=value,
                outside_training_range=bool(value < minimum or value > maximum)))
        coverage.append(dict(train_through=cutoff,validation_year=vy,evaluation_year=ey,
            training_elections=tr.election.nunique(),training_rows=len(tr),evaluation_rows=len(ev),
            missing_training_classes=','.join(p for p in PARTIES if p not in set(tr.winner)),
            unseen_incumbents=','.join(sorted(set(ev.incumbent)-set(tr.incumbent))),
            constant_numeric_features=','.join(c for c in config['features'] if pd.api.types.is_numeric_dtype(tr[c]) and tr[c].nunique()<=1),
            actual_changed_seats=int((ev.previous_winner.notna() & ev.winner.ne(ev.previous_winner)).sum()),
            missing_previous_winner=int(ev.previous_winner.isna().sum()),
            unseen_regions=','.join(sorted(set(ev['country/region'])-set(tr['country/region']))),
            unseen_region_rows=int((~ev['country/region'].isin(tr['country/region'])).sum())))
    for row in selected.itertuples():
        frame = predictions.loc[(predictions.split=='evaluation') & (predictions.train_through==row.train_through)
              & (predictions.architecture==row.architecture) & (predictions.learning_rate==row.learning_rate)
              & (predictions.patience==20)].sort_values('source_row')
        ev = data.loc[frame.source_row].copy()
        assert (ev.winner.to_numpy()==frame.winner.to_numpy()).all()
        assert (ev.election==row.evaluation_year).all()
        prob = frame[pcols].to_numpy()
        pred = np.array(PARTIES)[prob.argmax(axis=1)]
        actual = ev.winner.to_numpy(); prev = ev.previous_winner.to_numpy()
        correct = pred==actual
        known = ev.previous_winner.notna().to_numpy()
        changed = known & (actual!=prev)
        held = known & ~changed
        called = known & (pred!=prev)
        key = dict(evaluation_year=row.evaluation_year,architecture=row.architecture,policy=row.policy)
        saved[(row.evaluation_year,row.policy,row.architecture)] = (ev.index.to_numpy(),correct,pred)
        for seed in config['seeds']:
            path = cache / f'{row.train_through}_{row.architecture}_lr{row.learning_rate:g}_seed{seed}.npz'
            if path.exists():
                with np.load(path,allow_pickle=False) as z:
                    seed_probs = z['evaluation'][config['patiences'].index(20)]
                seed_pred = np.array(PARTIES)[seed_probs.argmax(axis=1)]
                seed_correct = seed_pred == actual
                seed_rows.append(dict(**key,seed=seed,accuracy=seed_correct.mean(),
                    changed_accuracy=seed_correct[changed].mean(),
                    **{p+'_seats':int((seed_pred==p).sum()) for p in PARTIES}))
        rows.append(dict(**key, learning_rate=row.learning_rate,rows=len(ev), accuracy=correct.mean(),
            overall_changed_score=(correct.mean()+correct[changed].mean())/2,
            log_loss=row.evaluation_log_loss,brier=row.evaluation_brier_score,
            previous_winner_accuracy=(actual==prev).mean(),
            known_previous_rows=int(known.sum()),known_previous_accuracy=correct[known].mean(),
            previous_winner_known_accuracy=(actual[known]==prev[known]).mean(),
            unknown_previous_rows=int((~known).sum()),
            unknown_previous_accuracy=correct[~known].mean() if (~known).any() else np.nan,
            accuracy_gain_pp=100*(correct.mean()-(actual==prev).mean()),
            changed_seats=int(changed.sum()),correct_changes=int((correct&changed).sum()),
            changed_accuracy=correct[changed].mean(),called_changes=int(called.sum()),
            correct_change_precision=(correct&changed).sum()/called.sum() if called.any() else np.nan,
            false_changes_on_held_seats=int((called&~changed).sum()),
            wrong_destination_on_changed_seats=int((called&changed&~correct).sum()),
            held_seat_accuracy=correct[held].mean(),
            seat_count_absolute_error=sum(abs((pred==p).sum()-(actual==p).sum()) for p in PARTIES)))
        for status,mask in [('known',known),('missing',~known)]:
            if mask.any():
                for p in PARTIES:
                    history_party.append(dict(**key,previous_winner_status=status,party=p,
                        actual_seats=int(((actual==p)&mask).sum()),predicted_seats=int(((pred==p)&mask).sum())))
        for i,p in enumerate(PARTIES):
            a=actual==p; pr=pred==p
            parties.append(dict(**key,party=p,actual_seats=int(a.sum()),predicted_seats=int(pr.sum()),
                expected_seats=float(prob[:,i].sum()),seat_error=int(pr.sum()-a.sum()),
                recall=(pr&a).sum()/a.sum() if a.any() else np.nan,
                precision=(pr&a).sum()/pr.sum() if pr.any() else np.nan))
            for q in PARTIES:
                confusions.append(dict(**key,actual_party=p,predicted_party=q,seats=int((a&(pred==q)).sum())))
        for old in PARTIES:
            for new in PARTIES:
                mask=(prev==old)&(actual==new)
                if mask.any():
                    transitions.append(dict(**key,previous_party=old,actual_party=new,seats=int(mask.sum()),
                        correctly_predicted=int(correct[mask].sum()),accuracy=correct[mask].mean(),
                        predicted_previous_party=int((pred[mask]==old).sum())))
    detail=pd.DataFrame(rows); party=pd.DataFrame(parties); transition=pd.DataFrame(transitions)
    if seed_rows:
        seed_frame = pd.DataFrame(seed_rows)
        assert len(seed_frame) == len(selected)*len(config['seeds']), 'Incomplete seed diagnostics; rerun/resume fitting.'
        seed_frame.to_csv(OUT/'selected_seed_metrics.csv',index=False)
        seed_frame.groupby(['evaluation_year','architecture','policy']).agg(
            seed_accuracy_mean=('accuracy','mean'),seed_accuracy_sd=('accuracy','std'),
            seed_accuracy_min=('accuracy','min'),seed_accuracy_max=('accuracy','max'),
            seed_changed_accuracy_sd=('changed_accuracy','std')).reset_index().to_csv(OUT/'seed_stability.csv',index=False)
    for (year,policy,arch),(indices,correct,pred) in saved.items():
        if arch=='32_16': continue
        bi,bc,bp=saved[(year,policy,'32_16')]
        np.testing.assert_array_equal(indices,bi)
        paired.append(dict(evaluation_year=year,policy=policy,architecture=arch,
            disagreement_seats=int((pred!=bp).sum()),challenger_only_correct=int((correct&~bc).sum()),
            baseline_only_correct=int((bc&~correct).sum()),net_correct_seats=int(correct.sum()-bc.sum())))
    for name,frame in [('election_diagnostics',detail),('per_party_recall',party),('transitions',transition),
                       ('confusion_matrices',pd.DataFrame(confusions)),('paired_architecture_comparison',pd.DataFrame(paired)),
                       ('window_coverage',pd.DataFrame(coverage)),('history_coverage_party_counts',pd.DataFrame(history_party)),('polling_feature_ranges',pd.DataFrame(feature_ranges))]:
        frame.to_csv(OUT/f'{name}.csv',index=False)
    # Descriptive sensitivity checks: elections, rather than seeds or seats,
    # are the units for assessing whether a pooled ranking is broadly supported.
    sensitivity = []
    years = sorted(detail.evaluation_year.unique())
    for omitted in [None, *years]:
        subset = detail if omitted is None else detail.loc[detail.evaluation_year != omitted]
        for (policy, arch), group in subset.groupby(['policy', 'architecture'], sort=False):
            sensitivity.append(dict(omitted_election='none' if omitted is None else str(omitted),
                policy=policy, architecture=arch, elections=len(group),
                mean_accuracy=group.accuracy.mean(), mean_log_loss=group.log_loss.mean(),
                mean_overall_changed_score=group.overall_changed_score.mean(),
                mean_known_history_gain_pp=100*(group.known_previous_accuracy-group.previous_winner_known_accuracy).mean(),
                elections_beating_persistence=int((group.known_previous_accuracy > group.previous_winner_known_accuracy).sum())))
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity['accuracy_rank'] = sensitivity.groupby(['omitted_election','policy']).mean_accuracy.rank(method='min',ascending=False)
    sensitivity['log_loss_rank'] = sensitivity.groupby(['omitted_election','policy']).mean_log_loss.rank(method='min')
    sensitivity['overall_changed_rank'] = sensitivity.groupby(['omitted_election','policy']).mean_overall_changed_score.rank(method='min',ascending=False)
    sensitivity.to_csv(OUT/'leave_one_election_out.csv',index=False)
    selected_rates = policies.loc[policies.policy.isin(['validation_all_rates','validation_midrange'])].copy()
    rate_comparison = selected_rates.loc[selected_rates.policy=='validation_all_rates'].merge(
        selected_rates.loc[selected_rates.policy=='validation_midrange'],
        on=['architecture','patience','train_through','evaluation_year'],suffixes=('_full','_midrange'))
    rate_comparison['same_rate'] = rate_comparison.learning_rate_full == rate_comparison.learning_rate_midrange
    rate_comparison['midrange_accuracy_delta_pp'] = 100*(rate_comparison.evaluation_accuracy_midrange-rate_comparison.evaluation_accuracy_full)
    rate_comparison['midrange_log_loss_delta'] = rate_comparison.evaluation_log_loss_midrange-rate_comparison.evaluation_log_loss_full
    rate_comparison.to_csv(OUT/'midrange_search_comparison.csv',index=False)
    # Readable figures use every evaluation election, not hard-coded original windows.
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True)
    for policy,axs in zip(['fixed_0.3','validation_all_rates'],axes.T):
        for arch in ARCHS:
            g=detail.loc[(detail.policy==policy)&(detail.architecture==arch)].sort_values('evaluation_year')
            axs[0].plot(g.evaluation_year,100*g.accuracy,marker='o',label=LABELS[arch])
            axs[1].plot(g.evaluation_year,100*g.changed_accuracy,marker='o',label=LABELS[arch])
        b=baseline.loc[baseline.baseline=='previous_winner'].sort_values('evaluation_year')
        axs[0].plot(b.evaluation_year,100*b.accuracy,'k--',label='Previous winner (missing = wrong)')
        axs[0].set_title('Fixed rate 0.3' if policy=='fixed_0.3' else 'Validation-selected rate')
        axs[0].set_ylabel('All-seat accuracy (%)'); axs[1].set_ylabel('Changed-seat accuracy (%)')
        for ax in axs:
            ax.set_xticks(b.evaluation_year);ax.tick_params(axis='x',rotation=45);ax.grid(alpha=.25)
        axs[0].legend(fontsize=8)
    fig.suptitle('Ten-seed ensembles · patience 20 · all evaluation elections')
    fig.tight_layout();fig.savefig(OUT/'architecture_by_election.png',dpi=160);plt.close(fig)
    focus_years = [year for year in [1997,2010] if year in set(detail.evaluation_year)]
    fig,axes=plt.subplots(1,len(focus_years),figsize=(5*len(focus_years),4),squeeze=False)
    axes = axes.ravel()
    for ax,year in zip(axes,focus_years):
        p=party.loc[(party.evaluation_year==year)&(party.policy=='validation_all_rates')]
        x=np.arange(len(PARTIES));width=.15
        actual=p.drop_duplicates('party').set_index('party').loc[PARTIES,'actual_seats']
        ax.bar(x-2*width,actual,width,label='Actual',color='0.35')
        for i,arch in enumerate(ARCHS):
            g=p.loc[p.architecture==arch].set_index('party').loc[PARTIES]
            ax.bar(x+(i-1)*width,g.predicted_seats,width,label=LABELS[arch],color=f'C{i}')
        ax.set_xticks(x,PARTIES);ax.set_title(str(year));ax.set_ylabel('Seats in supplied data')
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'government_change_seats.png',dpi=160);plt.close(fig)
    sections=['# All-election numerical results, 1997–2019\n',
        'Generated by `summarize_all_elections.py`. All architecture comparisons below use ten-seed ensembles and patience 20. See [the main report](../REPORT_all_elections_progress.md) for interpretation.\n',
        '## Temporal coverage\n',
        'Rows with missing previous winners remain in all-seat metrics but are excluded from change/hold diagnostics. The all-row previous-winner baseline counts missing predictions as incorrect; the known-previous subset is the fair direct comparison.\n',table(pd.DataFrame(coverage)),
        '\n## Architecture and learning-rate policy\n',
        'Fixed 0.3 holds the rate constant across architectures. Validation-selected rates compare complete procedures, so their differences include both architecture and rate. Election means give each election equal weight. The overall/changed score averages overall and changed-seat accuracy, matching the existing pipeline selector\'s equal weighting up to a factor of two. It is reported descriptively; rates remain selected by validation log loss.\n']
    for policy in ['fixed_0.3','validation_all_rates']:
        d=detail.loc[detail.policy==policy]
        sections.extend([f'\n### {policy}\n',table(d.groupby('architecture',sort=False).agg(
            mean_accuracy=('accuracy','mean'),mean_changed_accuracy=('changed_accuracy','mean'),
            mean_overall_changed_score=('overall_changed_score','mean'),
            mean_log_loss=('log_loss','mean'),mean_seat_count_absolute_error=('seat_count_absolute_error','mean')).reset_index(),
            percent=('mean_accuracy','mean_changed_accuracy')),
            '\nAccuracy by election:\n',table(d.pivot(index='evaluation_year',columns='architecture',values='accuracy').reset_index(),percent=ARCHS),
            '\nChanged-seat accuracy by election:\n',table(d.pivot(index='evaluation_year',columns='architecture',values='changed_accuracy').reset_index(),percent=ARCHS)])
    sections.extend(['\n![Architecture comparison](architecture_by_election.png)\n'])
    for year in sorted(detail.evaluation_year.unique()):
        sections.extend([f'\n## {year}\n',table(detail.loc[detail.evaluation_year==year,[
            'architecture','policy','learning_rate','accuracy','log_loss','previous_winner_accuracy','known_previous_accuracy','previous_winner_known_accuracy','unknown_previous_rows','changed_seats',
            'correct_changes','false_changes_on_held_seats','seat_count_absolute_error']],percent=('accuracy','previous_winner_accuracy','known_previous_accuracy','previous_winner_known_accuracy'))])
        p=party.loc[(party.evaluation_year==year)&(party.policy=='validation_all_rates')]
        seat=p.pivot(index='party',columns='architecture',values='predicted_seats').reindex(PARTIES)
        seat.insert(0,'actual',p.drop_duplicates('party').set_index('party').actual_seats)
        sections.extend(['\nValidation-selected predicted seat totals:\n',table(seat.reset_index())])
        if year in [1997,2010]:
            t=transition.loc[(transition.evaluation_year==year)&(transition.policy=='validation_all_rates') & transition.previous_party.ne(transition.actual_party)]
            sections.extend(['\nParty-to-party changes (whole dataset, including boundary mappings):\n',table(t[['architecture','previous_party','actual_party','seats','correctly_predicted','predicted_previous_party']])])
    sections.extend(['\n![Party seat totals](government_change_seats.png)\n','\n## Patience sensitivity\n',
        table(policies.loc[policies.policy=='validation_all_rates'].groupby(['architecture','patience'],sort=False).agg(
            mean_accuracy=('evaluation_accuracy','mean'),mean_changed_accuracy=('evaluation_changed_seat_accuracy','mean'),
            mean_log_loss=('evaluation_log_loss','mean')).reset_index(),percent=('mean_accuracy','mean_changed_accuracy')),
        '\n## Selection stability\n'])
    screen=pd.read_csv(OUT/'screening_policies.csv')
    a=screen.loc[(screen.policy=='validation_all_rates')&(screen.patience.astype(str)=='20')]
    b=selected.loc[selected.policy=='validation_all_rates']
    stability=a[['evaluation_year','architecture','learning_rate']].merge(b[['evaluation_year','architecture','learning_rate']],on=['evaluation_year','architecture'],suffixes=('_3seed','_10seed'))
    sections.extend([table(stability),'\n## Paired predictions versus 32/16\n',table(pd.DataFrame(paired))])
    if (OUT/'seed_stability.csv').exists():
        sections.extend(['\n## Seed variability\n',
            'Individual-seed accuracy spread measures optimisation sensitivity, not uncertainty over future elections. Ensemble accuracy need not equal mean individual accuracy.\n',
            table(pd.read_csv(OUT/'seed_stability.csv'),percent=('seed_accuracy_mean','seed_accuracy_sd','seed_accuracy_min','seed_accuracy_max','seed_changed_accuracy_sd'))])
    records = pd.read_csv(OUT/'confirmation_training_records.csv')
    sections.extend(['\n## Network size and stopping\n',table(records.loc[records.patience==20].groupby(['train_through','architecture'],sort=False).agg(
        parameters=('parameters','first'),median_best_epoch=('best_epoch','median'),max_best_epoch=('best_epoch','max'),
        capped_runs=('hit_epoch_cap','sum')).reset_index()),
        '\n## Polling support in training\n',
        'An out-of-range input identifies extrapolation, not proof that it caused an error. Poll columns are constant within each election.\n',
        table(pd.DataFrame(feature_ranges),percent=('training_min','training_max','evaluation_value')),
        '\n## Learning-rate sensitivity\n',table(results.loc[results.patience==20].groupby(['architecture','learning_rate'],sort=False).agg(
        mean_validation_loss=('validation_log_loss','mean'),mean_accuracy=('evaluation_accuracy','mean'),
        mean_log_loss=('evaluation_log_loss','mean')).reset_index(),percent=('mean_accuracy',))])
    sections.extend(['\n## Sensitivity to individual elections\n',
        'Each row omits one evaluation election from the descriptive average; no models are refitted or selected again. These overlapping summaries are sensitivity checks, not independent tests. Persistence comparisons use only rows with a known previous winner.\n',
        table(sensitivity,percent=('mean_accuracy',)),
        '\n## Does the smaller learning-rate search retain the full-search choice?\n',
        'The original three-window study found that rates {0.1, 0.2, 0.3, 0.5} retained the selected 32/16 rates. The following checks that claim across the expanded periods. Positive accuracy deltas favour the smaller search; positive log-loss deltas favour the full search.\n',
        table(rate_comparison[['evaluation_year','architecture','patience','learning_rate_full','learning_rate_midrange','same_rate','midrange_accuracy_delta_pp','midrange_log_loss_delta']])])
    (OUT/'RESULTS.md').write_text('\n\n'.join(sections)+'\n')
    print(detail.to_string(index=False))


if __name__=='__main__': main()
