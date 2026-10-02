"""Write a data dictionary for the actual train/test export schema."""
from pathlib import Path


DESCRIPTIONS = {
    'constituency_name': ('Metadata', 'Constituency name on the current election boundaries.'),
    'country/region': ('Predictor', 'Constituency country or English region, using standardised labels.'),
    'election': ('Metadata', 'Current election year; used for temporal train/validation/test splits.'),
    'constituency_id': ('Metadata', 'Constituency identifier on the current boundaries.'),
    'boundary_set': ('Metadata', 'Boundary period used for the constituency result.'),
    'election_type': ('Metadata', 'Actual or notional result; exported current rows are actual.'),
    'previous_election': ('Metadata', 'Comparison election, possibly a notional result on revised boundaries.'),
    'winner': ('Target', 'Current winning party category.'),
    'winning_party_vote_share': ('Current outcome', 'Largest current constituency party vote share.'),
    'second_party_vote_share': ('Current outcome', 'Second-largest current constituency party vote share; ties count as separate places.'),
    'previous_winner': ('Predictor', 'Previous constituency winning party: the holder.'),
    'previous_winning_party_last_election_vote_share': ('Predictor', 'Holder previous vote share. Reused for holder_previous_vote_share; no duplicate column.'),
    'previous_second_party_last_election_vote_share': ('Predictor', 'Challenger previous vote share (previous runner-up). Reused for challenger_previous_vote_share; no duplicate column.'),
    'previous_margin_1st_2nd': ('Predictor', 'Previous winning vote share minus previous runner-up vote share.'),
    'incumbent': ('Predictor', 'Party in national government before this election; distinct from constituency holder.'),
    'supported_incumbent': ('Predictor', '1 when previous constituency winner equals national incumbent, otherwise 0 (including unmatched previous winners).'),
    'incumbent_polling': ('Predictor', 'Labour polling when national incumbent is Labour; otherwise Conservative polling.'),
    'opposition_polling': ('Predictor', 'Conservative polling when national incumbent is Labour; otherwise Labour polling.'),
}
for party, label in {'con': 'Conservative', 'lab': 'Labour', 'lib': 'Liberal/Liberal Democrat', 'natSW': 'SNP/Plaid Cymru', 'oth': 'Other parties'}.items():
    DESCRIPTIONS[f'{party}_share'] = ('Current outcome', f'{label} current constituency vote share.')
    DESCRIPTIONS[f'previous_{party}_share'] = ('Predictor', f'{label} constituency share in the selected previous actual/notional result.' + (' Missing nationalist shares are replaced with zero in the previous-result lookup.' if party == 'natSW' else ''))
    DESCRIPTIONS[f'{party}_national_vote_share'] = ('Current outcome', f'{label} actual election votes divided by all valid votes across retained Great Britain constituencies. Unavailable for 2024.')
    DESCRIPTIONS[f'previous_{party}_national_vote_share'] = ('Predictor', f'{label} national vote share in the previous actual election, including when constituency comparisons use notional results.')
    DESCRIPTIONS[f'{party}_polling'] = ('Predictor', f'{label} national pre-election polling from the source row immediately before the GE marker.')
    DESCRIPTIONS[f'{party}_national_change'] = ('Predictor', f'{party}_polling minus previous_{party}_national_vote_share.' if party in ('con', 'lab', 'lib') else 'Missing: this party category has no polling input.')
    DESCRIPTIONS[f'projected_{party}_share'] = ('Predictor', f'Projected constituency share: previous_{party}_share + {party}_national_change. Adds the national vote-share change to the previous local share; not clipped to 0–1.')
for role in ('holder', 'challenger'):
    meaning = 'previous constituency winner' if role == 'holder' else 'previous constituency runner-up'
    DESCRIPTIONS[f'{role}_polling'] = ('Predictor', f'Current national polling for the {meaning}; missing without a supported party match/polling.')
    DESCRIPTIONS[f'{role}_national_swing'] = ('Predictor', f'National change for the {meaning}: current polling minus previous actual national vote share. Missing without a supported party match/polling.')


def write_predictor_guide(train_test_data, path: Path):
    columns = list(dict.fromkeys(column for frame in train_test_data.values() for column in frame.columns))
    missing = set(columns) - DESCRIPTIONS.keys()
    if missing:
        raise ValueError(f'Add predictor descriptions for new export columns: {sorted(missing)}')
    lines = [
        '# Train/test predictor guide', '',
        'Generated with train.csv and test.csv. This lists every exported column; models use their own explicit feature lists, so exported predictors are not automatically selected.', '',
        'Shares, polling and changes use fractions: 0.05 means five percentage points. National results exclude Ireland and Northern Ireland. Missing values export as empty CSV fields.', '',
        'Current outcomes and the target are not pre-election predictors. Election and identifiers are metadata. Party rankings follow the categories available in each cleaner; some notional sources rank only con, lib, lab and natSW.', '',
        'Holder means previous constituency winner; challenger means previous runner-up. Their previous vote shares reuse the existing winning/second-place columns. Polling is available only for con, lab and lib. For runner-up ties, eligible parties are checked in con, lib, lab, natSW order, excluding the holder. Unsupported or unmatched runner-up categories have missing polling/swing.', '',
        '| Column | Role | Description |', '| --- | --- | --- |',
    ]
    for column in columns:
        role, description = DESCRIPTIONS[column]
        lines.append(f'| `{column}` | {role} | {description} |')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
