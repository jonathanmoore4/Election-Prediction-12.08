"""Forecast 2019 from pre-2019 results and polling, without evaluating 2019."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.stats import t
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from Models.xgboost_model import FEATURE_COLUMNS

PARTIES = ['Labour', 'Conservative', 'Liberal Democrat', 'Other', 'natSW']
LABELS = {'lab': 'Labour', 'con': 'Conservative', 'lib': 'Liberal Democrat',
          'oth': 'Other', 'natSW': 'natSW'}
POLL_COLUMNS = {'Labour': 'Labour', 'Conservative': 'Conservative',
                'Liberal Democrat': 'LD'}
IDENTIFIERS = ['election', 'constituency_id', 'constituency_name', 'country/region']


def load_inputs(path):
    """Project the forecast onto predictors; retain labels only through 2017.

    The CSV contains multiple elections. Filter each label chunk immediately;
    no 2019 outcome is aggregated, inspected, returned, or used as a predictor.
    """
    columns = list(dict.fromkeys(IDENTIFIERS + FEATURE_COLUMNS))
    predictors = pd.read_csv(path, usecols=columns)
    forecast = predictors.loc[predictors.election.eq(2019)].copy().reset_index(drop=True)
    historical_chunks = []
    for chunk in pd.read_csv(path, usecols=columns + ['winner'], chunksize=512):
        historical_chunk = chunk.loc[chunk.election.le(2017)].copy()
        if not historical_chunk.empty:
            historical_chunks.append(historical_chunk)
    if not historical_chunks:
        raise ValueError('Historical training rows are required.')
    historical = pd.concat(historical_chunks, ignore_index=True)
    if historical.empty or forecast.empty:
        raise ValueError('Historical training rows and 2019 predictors are required.')
    for frame in (historical, forecast):
        if frame.constituency_id.isna().any() or frame.duplicated(['election', 'constituency_id']).any():
            raise ValueError('Each election must have unique, nonmissing constituency IDs.')
    if historical.winner.isna().any() or set(historical.winner) != set(LABELS):
        raise ValueError('Historical labels must contain exactly the five expected parties.')
    return historical, forecast


def fit_probabilities(historical, forecast):
    """Match notebook 03's predictors, dummy encoding, and accuracy grid search.

    Retain Other to meet the five-class forecast requirement. All tuning and
    refitting use elections through 2017; no 2019 labels or eval_set are supplied.
    """
    if not historical.election.le(2017).all():
        raise ValueError('Training must stop at 2017.')
    X_train = pd.get_dummies(historical[FEATURE_COLUMNS])
    X_2019 = pd.get_dummies(forecast[FEATURE_COLUMNS]).reindex(columns=X_train.columns, fill_value=0)
    encoder = LabelEncoder()
    y = encoder.fit_transform(historical.winner)
    search = GridSearchCV(
        XGBClassifier(objective='multi:softprob', eval_metric='mlogloss',
                      n_jobs=1, random_state=42),
        {'n_estimators': [20, 35, 50, 100], 'max_depth': [2, 3, 4, 5]},
        scoring='accuracy', cv=5, n_jobs=-1, error_score='raise',
    )
    search.fit(X_train, y)
    class_names = [LABELS[label] for label in encoder.inverse_transform(search.classes_.astype(int))]
    probabilities = pd.DataFrame(search.predict_proba(X_2019), columns=class_names)[PARTIES]
    return probabilities, search, encoder


def polling_seat_ranges(historical, forecast):
    """Separate OLS fits, with Student-t prediction intervals for a new election."""
    if not historical.election.le(2017).all():
        raise ValueError('Polling regressions must stop at 2017.')
    for frame in (historical, forecast):
        polling = frame[list(POLL_COLUMNS.values())]
        if not np.isfinite(polling.to_numpy()).all():
            raise ValueError('Polling must be finite.')
        if not frame.groupby('election')[list(POLL_COLUMNS.values())].nunique().eq(1).all().all():
            raise ValueError('Expected one national polling value per party and election.')
    seats = pd.crosstab(historical.election, historical.winner.map(LABELS)).reindex(columns=PARTIES, fill_value=0)
    polls = historical.groupby('election')[list(POLL_COLUMNS.values())].first()
    records, fits, history = [], {}, []
    for party, column in POLL_COLUMNS.items():
        x = polls[column].to_numpy(dtype=float)
        y = seats.loc[polls.index, party].to_numpy(dtype=float)
        n = len(x)
        sxx = np.sum((x - x.mean()) ** 2)
        if n < 3 or sxx <= 0:
            raise ValueError('OLS prediction intervals require >=3 elections and varying polling.')
        model = LinearRegression().fit(x[:, None], y)
        x_new = float(forecast[column].iloc[0])
        predicted = float(model.predict([[x_new]])[0])
        residual_variance = np.sum((y - model.predict(x[:, None])) ** 2) / (n - 2)
        # The leading 1 is the new-election error, absent from a mean-response CI.
        prediction_se = np.sqrt(residual_variance * (1 + 1 / n + (x_new - x.mean()) ** 2 / sxx))
        half_width = t.ppf(0.975, df=n - 2) * prediction_se
        lower, upper = predicted - half_width, predicted + half_width
        # Integers inside the continuous interval, intersected with feasible seats.
        lower_seats = max(0, int(np.ceil(lower)))
        upper_seats = min(len(forecast), int(np.floor(upper)))
        if lower_seats > upper_seats:
            raise ValueError(f'No feasible integer seats in the prediction interval for {party}.')
        records.append(dict(party=party, polling_2019=x_new, predicted_seats=predicted,
                            prediction_lower_95=lower, prediction_upper_95=upper,
                            lower_seats=lower_seats, upper_seats=upper_seats,
                            historical_elections=n, residual_df=n - 2))
        fits[party] = model
        history.extend(dict(election=int(year), party=party, polling=float(poll), seats=int(count))
                       for year, poll, count in zip(polls.index, x, y))
    return pd.DataFrame(records).set_index('party'), fits, pd.DataFrame(history)


def assign_constituencies(probabilities, ranges):
    """Solve the complete binary assignment with HiGHS via scipy.optimize.milp."""
    p = probabilities[PARTIES].to_numpy(dtype=float)
    if len(p) == 0 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Probabilities must be finite and between zero and one.')
    if not np.allclose(p.sum(axis=1), 1, atol=1e-6):
        raise ValueError('Each constituency probability row must sum to one.')
    n, k = p.shape
    lower = ranges.loc[list(POLL_COLUMNS), 'lower_seats'].to_numpy(dtype=float)
    upper = ranges.loc[list(POLL_COLUMNS), 'upper_seats'].to_numpy(dtype=float)
    if (not np.isfinite(np.r_[lower, upper]).all() or
            not np.equal(np.r_[lower, upper], np.r_[lower, upper].round()).all() or
            (lower < 0).any() or (upper > n).any() or (lower > upper).any() or lower.sum() > n):
        raise ValueError('Polling seat bounds are infeasible or noninteger; no automatic relaxation.')
    # Row-major variable order: one variable per (constituency, party).
    rows = [np.repeat(np.arange(n), k)]
    cols = [np.arange(n * k)]
    for offset, party in enumerate(POLL_COLUMNS):
        rows.append(np.full(n, n + offset))
        cols.append(np.arange(n) * k + PARTIES.index(party))
    row_indices, col_indices = np.concatenate(rows), np.concatenate(cols)
    matrix = coo_matrix((np.ones(len(row_indices)), (row_indices, col_indices)),
                        shape=(n + 3, n * k)).tocsc()
    costs = -np.log(np.clip(p, 1e-15, 1)).ravel()
    solution = milp(c=costs, integrality=np.ones(n * k), bounds=Bounds(0, 1),
                    constraints=LinearConstraint(matrix, np.r_[np.ones(n), lower],
                                                 np.r_[np.ones(n), upper]),
                    options={'mip_rel_gap': 0.0})
    if not solution.success or solution.status != 0:
        raise RuntimeError(f'No proven optimal assignment: {solution.message}')
    raw = solution.x.reshape(n, k)
    assigned = np.rint(raw).astype(int)
    if not np.allclose(raw, assigned, atol=1e-6) or not (assigned.sum(axis=1) == 1).all():
        raise RuntimeError('Solver returned an invalid binary assignment.')
    totals = assigned.sum(axis=0)
    for party in POLL_COLUMNS:
        if not ranges.loc[party, 'lower_seats'] <= totals[PARTIES.index(party)] <= ranges.loc[party, 'upper_seats']:
            raise RuntimeError('Solver returned an assignment outside the seat bounds.')
    winners = np.asarray(PARTIES)[assigned.argmax(axis=1)]
    return winners, solution


def run_forecast(data_path, output_dir=None):
    historical, forecast = load_inputs(data_path)
    probabilities, search, encoder = fit_probabilities(historical, forecast)
    # Store probabilities before creating any winners.
    probability_frame = pd.concat([
        forecast[IDENTIFIERS], probabilities.add_suffix(' probability')
    ], axis=1)
    ranges, polling_models, polling_history = polling_seat_ranges(historical, forecast)
    constrained, solution = assign_constituencies(probabilities, ranges)
    final = probability_frame.copy()
    final['unconstrained_winner'] = probabilities.idxmax(axis=1)
    final['constrained_winner'] = constrained
    totals = pd.DataFrame({
        'unconstrained_seats': final.unconstrained_winner.value_counts().reindex(PARTIES, fill_value=0),
        'constrained_seats': final.constrained_winner.value_counts().reindex(PARTIES, fill_value=0),
    }).rename_axis('party')
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        probability_frame.to_csv(output_dir / '2019_probabilities.csv', index=False)
        final.to_csv(output_dir / '2019_constituency_predictions.csv', index=False)
        totals.to_csv(output_dir / '2019_seat_totals.csv')
        ranges.to_csv(output_dir / '2019_polling_seat_ranges.csv')
        polling_history.to_csv(output_dir / 'historical_polling_seats.csv', index=False)
    return dict(predictions=final, probabilities=probability_frame, seat_totals=totals,
                seat_ranges=ranges, polling_history=polling_history,
                polling_models=polling_models, grid_search=search, label_encoder=encoder,
                optimisation=solution, training_elections=sorted(historical.election.unique().tolist()))


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    result = run_forecast(root / 'TEST_TRAIN/train.csv', root / 'Analysis and model development/outputs/constrained_2019')
    print('Training elections:', result['training_elections'])
    print('Selected parameters:', result['grid_search'].best_params_)
    print(result['seat_ranges'].to_string())
    print(result['seat_totals'].to_string())
    print('Optimiser:', result['optimisation'].message)
