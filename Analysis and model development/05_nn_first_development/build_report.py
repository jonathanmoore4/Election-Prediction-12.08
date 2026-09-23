"""Rebuild the report using only the seven retained evaluation elections."""
from pathlib import Path
import base64
import re
import pandas as pd
import mistune
from summarize_results import table

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'


def main():
    d=pd.read_csv(OUT/'election_diagnostics.csv')
    assert set(d.evaluation_year)=={1997,2001,2005,2010,2015,2017,2019}
    selected=d.loc[d.policy=='validation_all_rates'].copy()
    parties=pd.read_csv(OUT/'per_party_recall.csv')
    policies=pd.read_csv(OUT/'confirmation_policies.csv')
    sensitivity=pd.read_csv(OUT/'leave_one_election_out.csv',keep_default_na=False)
    rates=pd.read_csv(OUT/'midrange_search_comparison.csv')
    summaries=d.groupby(['policy','architecture'],sort=False).agg(mean_accuracy=('accuracy','mean'),mean_changed_accuracy=('changed_accuracy','mean'),mean_score=('overall_changed_score','mean'),mean_log_loss=('log_loss','mean'),mean_absolute_seat_error=('seat_count_absolute_error','mean')).reset_index()
    selected_summary=summaries.loc[summaries.policy=='validation_all_rates']
    fixed_summary=summaries.loc[summaries.policy=='fixed_0.3']
    winner=selected_summary.loc[selected_summary.mean_accuracy.idxmax()]
    fixed_winner=fixed_summary.loc[fixed_summary.mean_score.idxmax()]
    sections=['# First neural-network development: elections through 2019\n',
        '**Completed scope: seven evaluation elections—1997, 2001, 2005, 2010, 2015, 2017 and 2019.** Every table, figure, average and ranking in this package is restricted to that period. Earlier elections from 1987 supply training and validation history.\n',
        '## Findings and candidate models\n',
        f'With validation-selected rates, **{winner.architecture.replace("_","/")} has the highest mean overall accuracy ({winner.mean_accuracy:.2%})**. At fixed rate 0.3, **{fixed_winner.architecture.replace("_","/")} has the highest mean equally weighted overall/changed-seat score ({fixed_winner.mean_score:.2%})**. These rankings were recomputed from the seven retained elections.\n\n'
        'The two candidate procedures are now implemented alongside NN01 in the pipeline. They remain retrospective development candidates: resizing a network does not consistently fix missed changes, false changes or biased party totals. The objective matters—overall accuracy, changed-seat recall and probability quality do not always favour the same procedure.\n',
        '| Pipeline model | Hidden layers | Learning-rate policy | Notebook |\n| --- | --- | --- | --- |\n'
        '| NN01 | 32/16 | Validation-selected from 0.1, 0.2, 0.3, 0.5 | Existing reference implementation |\n'
        '| NN02 | 64/32 | Validation-selected from 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0 | [NN02 notebook](01_NN02_64_32_validation_selected.ipynb) |\n'
        '| NN03 | 16 | Fixed 0.3 | [NN03 notebook](02_NN03_16_fixed_0_3.ipynb) |\n\n'
        'All use ten seeded networks, SGD, cross-entropy, patience 20 and restoration of each seed’s best validation checkpoint. The latest supplied whole election is held out for validation. NN02 selects the rate with the lowest **ensemble** validation log loss; NN03 holds its rate fixed while retaining validation-based early stopping. The study’s full-grid 32/16 comparator is distinct from NN01’s smaller four-rate grid.\n',
        '## Experiment and interpretation\n',
        'Each window trains through a cutoff, validates on the next whole election and evaluates on the following one. Preprocessing is fitted on training rows only. Four architectures, eight rates and ten seeds give **2,240 trajectories**, supporting patience 10/20/50, **6,720 seed/patience records**, and **672 ten-seed ensembles**. The three-seed screening summaries reuse the first three seeds and are not independent replications.\n\n'
        'These are already-inspected historical development results. Restricting the reporting period does not create an independent test. Elections overlap across training windows; ten seeds measure optimisation variation, not ten independent elections. The upstream features and boundary mappings have not received a fresh historical-availability audit.\n',
        '## Architecture comparisons\n',
        'Each election receives equal weight. The overall/changed score averages overall accuracy and accuracy on changed seats, matching the pipeline selector up to a factor of two. Higher accuracy/score is better; lower log loss and absolute party-seat error is better. The fixed-rate comparison holds the rate constant, whereas selected-rate comparisons include tuning effects.\n']
    for policy,title in [('validation_all_rates','Validation-selected rates'),('fixed_0.3','Fixed learning rate 0.3')]:
        sections += [f'\n### {title}\n',table(summaries.loc[summaries.policy==policy].drop(columns='policy'),percent=('mean_accuracy','mean_changed_accuracy','mean_score'))]
    sections += ['\n## Every election\n\nOverall accuracy with validation-selected rates:\n',table(selected.pivot(index='evaluation_year',columns='architecture',values='accuracy').reset_index(),percent=('16','32','32_16','64_32')),
        '\nCorrect new party among seats whose known previous winner changes:\n',table(selected.pivot(index='evaluation_year',columns='architecture',values='changed_accuracy').reset_index(),percent=('16','32','32_16','64_32')),
        '\n![Architecture comparisons](results/architecture_by_election.png)\n',
        '\n## The two pipeline candidates\n',
        'These rows reproduce NN02’s and NN03’s respective study procedures. They show why one candidate should not be declared universally best.\n',
        table(d.loc[((d.architecture=='64_32')&(d.policy=='validation_all_rates'))|((d.architecture=='16')&(d.policy=='fixed_0.3')),['evaluation_year','architecture','policy','learning_rate','accuracy','changed_accuracy','overall_changed_score','log_loss']],percent=('accuracy','changed_accuracy','overall_changed_score')),
        '\n## Comparison with predicting the previous winner\n',
        'The baseline and network are compared on exactly the same known-history rows. The 91 missing-history rows in 1997 remain in all-seat accuracy but are excluded from this comparison and from change/hold diagnostics. Correct changes minus false changes on held seats equals the net additional correct predictions against persistence.\n']
    selected['gain_pp']=100*(selected.known_previous_accuracy-selected.previous_winner_known_accuracy)
    sections += [table(selected.groupby('architecture',sort=False).agg(mean_known_accuracy=('known_previous_accuracy','mean'),mean_baseline=('previous_winner_known_accuracy','mean'),mean_gain_pp=('gain_pp','mean'),elections_beating_baseline=('gain_pp',lambda x:int((x>0).sum()))).reset_index(),percent=('mean_known_accuracy','mean_baseline')),
        '\n## Election-specific findings and party totals\n']
    notes={1997:'Training uses only 1987: national poll columns are constant, there are no positive other-party examples, and 91 evaluation rows lack previous winners. All architectures miss most changed seats and overpredict Conservatives.',2001:'Only 24 seats change winner. All selected architectures score highly overall but underperform predicting the previous winner.',2005:'Many changes are correctly identified, but false changes on held seats outweigh that benefit. Conservative totals are overpredicted and Labour totals underpredicted.',2010:'The 64/32 architecture improves overall accuracy to 89.24%, versus 87.34% for 32/16. All four architectures nevertheless overpredict Conservative totals.',2015:'The Liberal Democrat and combined nationalist totals show large shared errors across architectures. Architecture changes alone do not resolve this transfer failure.',2017:'High overall accuracy masks weak changed-seat recall. The transition tables distinguish no-change predictions from changes to the wrong destination party.',2019:'The full-grid 32/16 procedure reaches 90.19% accuracy, but predicts only one Liberal Democrat seat against eleven actual. Single-layer architectures improve at fixed rate 0.3 compared with their validation-selected rates.'}
    for year in sorted(selected.evaluation_year.unique()):
        g=parties.loc[(parties.evaluation_year==year)&(parties.policy=='validation_all_rates')]
        t=g.pivot(index='party',columns='architecture',values='predicted_seats').reindex(['con','lab','lib','natSW','oth'])
        t.insert(0,'actual',g.drop_duplicates('party').set_index('party').actual_seats)
        sections += [f'\n### {year}\n',notes[year],table(t.reset_index())]
    sections += ['\nCounts cover supplied Great Britain rows, not the entire UK Parliament. `natSW` combines SNP and Plaid Cymru, while `oth` pools other parties. Predicted totals use argmax winners and do not provide a parliamentary-majority probability.\n',
        '\n## Sensitivity to individual elections\n',
        'Each row counts first-place rankings after leaving out one of the seven evaluation elections. Models are not retrained and local rate selections remain fixed. Ties are counted for each tied architecture. This is descriptive sensitivity, not independent testing.\n']
    loo=sensitivity.loc[sensitivity.omitted_election.ne('none')]
    ranks=loo.groupby(['policy','architecture'],sort=False).agg(accuracy_first=('accuracy_rank',lambda x:int((x==1).sum())),score_first=('overall_changed_rank',lambda x:int((x==1).sum())),log_loss_first=('log_loss_rank',lambda x:int((x==1).sum())))
    sections += [table(ranks.reset_index()),'\n## Rates and patience\n\nSelected rates at patience 20:\n',table(selected.pivot(index='evaluation_year',columns='architecture',values='learning_rate').reset_index()),
        '\nFor 32/16, the smaller grid uses {0.1, 0.2, 0.3, 0.5}. Positive accuracy deltas favour the smaller grid; positive log-loss deltas favour the full grid.\n',table(rates.loc[(rates.architecture=='32_16')&(rates.patience.astype(str)=='20'),['evaluation_year','learning_rate_full','learning_rate_midrange','same_rate','midrange_accuracy_delta_pp','midrange_log_loss_delta']]),
        '\nPatience comparisons with validation-selected rates:\n',table(policies.loc[policies.policy=='validation_all_rates'].groupby(['architecture','patience'],sort=False).agg(mean_accuracy=('evaluation_accuracy','mean'),mean_changed_accuracy=('evaluation_changed_seat_accuracy','mean'),mean_log_loss=('evaluation_log_loss','mean')).reset_index(),percent=('mean_accuracy','mean_changed_accuracy')),
        '\n## Verification and reproduction\n',
        (OUT/'verification.txt').read_text(),
        '\n[Executed study notebook](00_architecture_study.ipynb) · [Full numerical results](results/RESULTS.md) · [Reproduction instructions](README.md)\n\nThe retained data, probabilities, diagnostics and scripts are in this directory. The new pipeline classes share NN01’s training implementation, with independent architecture/rate settings; the model notebooks use those same classes to avoid diverging implementations.\n']
    report='\n\n'.join(sections)+'\n'
    (HERE/'REPORT_all_elections_progress.md').write_text(report)
    html=mistune.create_markdown(plugins=['table'])(report)
    for src in re.findall(r'<img src="([^"]+)"',html):
        html=html.replace(src,'data:image/png;base64,'+base64.b64encode((HERE/src).read_bytes()).decode())
    style='body{font:16px/1.5 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#183047}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:right;overflow-wrap:anywhere}th{background:#eef3f7}img{max-width:100%}h2{margin-top:36px}@media print{body{font-size:10px;margin:0}table{font-size:8px}h2,h3{break-after:avoid}tr,img{break-inside:avoid}}'
    (HERE/'REPORT_all_elections_progress.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Neural-network development through 2019</title><style>'+style+'</style><body>'+html+'</body></html>')
    print('Published report restricted to seven elections through 2019.')


if __name__=='__main__':main()
