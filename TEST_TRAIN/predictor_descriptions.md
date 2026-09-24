# Train/test predictor guide

Generated with train.csv and test.csv. This lists every exported column; models use their own explicit feature lists, so exported predictors are not automatically selected.

Shares, polling and changes use fractions: 0.05 means five percentage points. National results exclude Ireland and Northern Ireland. Missing values export as empty CSV fields.

Current outcomes and the target are not pre-election predictors. Election and identifiers are metadata. Party rankings follow the categories available in each cleaner; some notional sources rank only con, lib, lab and natSW.

Holder means previous constituency winner; challenger means previous runner-up. Their previous vote shares reuse the existing winning/second-place columns. Polling is available only for con, lab and lib. For runner-up ties, eligible parties are checked in con, lib, lab, natSW order, excluding the holder. Unsupported or unmatched runner-up categories have missing polling/swing.

| Column | Role | Description |
| --- | --- | --- |
| `constituency_name` | Metadata | Constituency name on the current election boundaries. |
| `country/region` | Predictor | Constituency country or English region, using standardised labels. |
| `election` | Metadata | Current election year; used for temporal train/validation/test splits. |
| `winning_party_vote_share` | Current outcome | Largest current constituency party vote share. |
| `second_party_vote_share` | Current outcome | Second-largest current constituency party vote share; ties count as separate places. |
| `winner` | Target | Current winning party category. |
| `constituency_id` | Metadata | Constituency identifier on the current boundaries. |
| `boundary_set` | Metadata | Boundary period used for the constituency result. |
| `election_type` | Metadata | Actual or notional result; exported current rows are actual. |
| `con_share` | Current outcome | Conservative current constituency vote share. |
| `lib_share` | Current outcome | Liberal/Liberal Democrat current constituency vote share. |
| `lab_share` | Current outcome | Labour current constituency vote share. |
| `natSW_share` | Current outcome | SNP/Plaid Cymru current constituency vote share. |
| `previous_election` | Metadata | Comparison election, possibly a notional result on revised boundaries. |
| `con_national_vote_share` | Current outcome | Conservative actual election votes divided by all valid votes across retained Great Britain constituencies. Unavailable for 2024. |
| `previous_con_national_vote_share` | Predictor | Conservative national vote share in the previous actual election, including when constituency comparisons use notional results. |
| `lab_national_vote_share` | Current outcome | Labour actual election votes divided by all valid votes across retained Great Britain constituencies. Unavailable for 2024. |
| `previous_lab_national_vote_share` | Predictor | Labour national vote share in the previous actual election, including when constituency comparisons use notional results. |
| `lib_national_vote_share` | Current outcome | Liberal/Liberal Democrat actual election votes divided by all valid votes across retained Great Britain constituencies. Unavailable for 2024. |
| `previous_lib_national_vote_share` | Predictor | Liberal/Liberal Democrat national vote share in the previous actual election, including when constituency comparisons use notional results. |
| `natSW_national_vote_share` | Current outcome | SNP/Plaid Cymru actual election votes divided by all valid votes across retained Great Britain constituencies. Unavailable for 2024. |
| `previous_natSW_national_vote_share` | Predictor | SNP/Plaid Cymru national vote share in the previous actual election, including when constituency comparisons use notional results. |
| `oth_national_vote_share` | Current outcome | Other parties actual election votes divided by all valid votes across retained Great Britain constituencies. Unavailable for 2024. |
| `previous_oth_national_vote_share` | Predictor | Other parties national vote share in the previous actual election, including when constituency comparisons use notional results. |
| `previous_winning_party_last_election_vote_share` | Predictor | Holder previous vote share. Reused for holder_previous_vote_share; no duplicate column. |
| `previous_second_party_last_election_vote_share` | Predictor | Challenger previous vote share (previous runner-up). Reused for challenger_previous_vote_share; no duplicate column. |
| `previous_winner` | Predictor | Previous constituency winning party: the holder. |
| `previous_con_share` | Predictor | Conservative constituency share in the selected previous actual/notional result. |
| `previous_lib_share` | Predictor | Liberal/Liberal Democrat constituency share in the selected previous actual/notional result. |
| `previous_lab_share` | Predictor | Labour constituency share in the selected previous actual/notional result. |
| `previous_natSW_share` | Predictor | SNP/Plaid Cymru constituency share in the selected previous actual/notional result. Missing nationalist shares are replaced with zero in the previous-result lookup. |
| `con_polling` | Predictor | Conservative national pre-election polling from the source row immediately before the GE marker. |
| `lab_polling` | Predictor | Labour national pre-election polling from the source row immediately before the GE marker. |
| `lib_polling` | Predictor | Liberal/Liberal Democrat national pre-election polling from the source row immediately before the GE marker. |
| `incumbent` | Predictor | Party in national government before this election; distinct from constituency holder. |
| `supported_incumbent` | Predictor | 1 when previous constituency winner equals national incumbent, otherwise 0 (including unmatched previous winners). |
| `incumbent_polling` | Predictor | Labour polling when national incumbent is Labour; otherwise Conservative polling. |
| `opposition_polling` | Predictor | Conservative polling when national incumbent is Labour; otherwise Labour polling. |
| `con_national_change` | Predictor | con_polling minus previous_con_national_vote_share. |
| `lab_national_change` | Predictor | lab_polling minus previous_lab_national_vote_share. |
| `lib_national_change` | Predictor | lib_polling minus previous_lib_national_vote_share. |
| `natSW_national_change` | Predictor | Missing: this party category has no polling input. |
| `oth_national_change` | Predictor | Missing: this party category has no polling input. |
| `previous_margin_1st_2nd` | Predictor | Previous winning vote share minus previous runner-up vote share. |
| `projected_con_share` | Predictor | Projected constituency share: previous_con_share + con_national_change. Adds the national vote-share change to the previous local share; not clipped to 0–1. |
| `projected_lib_share` | Predictor | Projected constituency share: previous_lib_share + lib_national_change. Adds the national vote-share change to the previous local share; not clipped to 0–1. |
| `projected_lab_share` | Predictor | Projected constituency share: previous_lab_share + lab_national_change. Adds the national vote-share change to the previous local share; not clipped to 0–1. |
| `holder_national_swing` | Predictor | National change for the previous constituency winner: current polling minus previous actual national vote share. Missing without a supported party match/polling. |
| `challenger_national_swing` | Predictor | National change for the previous constituency runner-up: current polling minus previous actual national vote share. Missing without a supported party match/polling. |
| `holder_polling` | Predictor | Current national polling for the previous constituency winner; missing without a supported party match/polling. |
| `challenger_polling` | Predictor | Current national polling for the previous constituency runner-up; missing without a supported party match/polling. |
