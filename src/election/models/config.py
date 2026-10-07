"""Existing whole-election schedules; election years are never random folds."""

OUTER_ELECTIONS = (2005, 2010, 2015, 2017, 2019)
INNER_ELECTIONS = (1997, 2001, 2005, 2010, 2015, 2017, 2019)
PARTIES = ('con', 'lab', 'lib', 'natSW', 'oth')
SCORING_ID = 'winner-accuracy-equal-election-mean-v1'
TARGET_ID = 'winner-five-party-gb-v1'
EVALUATION_PROTOCOL = 'nested-election-optuna-random-inner-checkpoint-median-refit-v2'

MODEL_COMPARISON_SCHEDULE = {
    'name': 'historical-five-election-v1',
    'purpose': 'model_comparison',
    'outer_elections': list(OUTER_ELECTIONS),
    'inner_elections': {year: [v for v in INNER_ELECTIONS if v < year]
                        for year in OUTER_ELECTIONS},
    'training_cutoff': None,
}

FINAL_EVALUATION_SCHEDULE = {
    'name': 'final-2024-v1',
    'purpose': 'final_evaluation',
    'outer_elections': [2024],
    'inner_elections': {2024: list(INNER_ELECTIONS)},
    'training_cutoff': 2019,
}
