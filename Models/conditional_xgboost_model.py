"""Two conditional XGBoost stages exposed as one pipeline candidate.

Adapted from 06_xgboost_changed_extended_multilayer.ipynb. Both stages use
independent chronological accuracy searches. Role mapping and preprocessing are
fitted on earlier rows only, including when constructing fold-specific targets.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from Models.custom_model import custom_model
from Models.missing_data import ElectionImputer

PARTIES = ('con', 'lib', 'lab', 'natSW', 'oth')
ROLES = ('incumbent', 'contesting_party', 'third_party', 'fourth_party', 'fifth_party')
PREVIOUS_COLUMNS = [f'previous_{p}_share' for p in PARTIES[:-1]]
FAMILIES = {
    'share': PREVIOUS_COLUMNS + [None],
    'projected_share': ['projected_con_share', 'projected_lib_share', 'projected_lab_share', None, None],
    'polling': ['con_polling', 'lib_polling', 'lab_polling', None, None],
    'previous_national_share': [f'previous_{p}_national_vote_share' for p in PARTIES[:-1]] + [None],
}
FEATURE_COLUMNS = list(dict.fromkeys(
    ['previous_winner', 'country/region', 'incumbent',
     'previous_winning_party_last_election_vote_share']
    + [col for columns in FAMILIES.values() for col in columns if col]
))
PARAM_GRID = {'max_depth': [2, 3, 4, 5], 'n_estimators': [25, 50, 100, 200]}


class RoleMapper:
    """Impute ranking inputs before assigning all five parties to unique roles."""

    def fit(self, data):
        self.imputer = ElectionImputer().fit(data)
        data = self.imputer.transform(data)
        self.shares = SimpleImputer(strategy='median', keep_empty_features=True)
        self.shares.fit(data[PREVIOUS_COLUMNS])
        counts = data.previous_winner.dropna().value_counts()
        if counts.empty:
            raise ValueError('Role mapping needs at least one known previous winner.')
        self.previous_winner = counts.index[0]
        return self

    def transform(self, data):
        frame = self.imputer.transform(data)[FEATURE_COLUMNS].copy().reset_index(drop=True)
        frame['previous_winner'] = frame.previous_winner.fillna(self.previous_winner)
        incumbent = pd.Index(PARTIES).get_indexer(frame.previous_winner)
        if (incumbent < 0).any():
            raise ValueError('Previous winners must use con, lib, lab, natSW or oth.')
        if not len(frame):
            return pd.DataFrame(), np.empty((0, 5), dtype=int)
        shares = self.shares.transform(frame[PREVIOUS_COLUMNS])
        frame[PREVIOUS_COLUMNS] = shares
        # oth is last regardless of named-party shares, unless it is incumbent.
        ranked = np.column_stack([np.argsort(-shares, axis=1, kind='stable'),
                                  np.full(len(frame), 4)])
        challengers = ranked[ranked != incumbent[:, None]].reshape(-1, 4)
        roles = np.column_stack([incumbent, challengers])
        features = pd.DataFrame(index=frame.index)
        rows = np.arange(len(frame))
        for number, role in enumerate(ROLES):
            features[f'{role}_party'] = np.asarray(PARTIES)[roles[:, number]]
            for suffix, columns in FAMILIES.items():
                values = np.column_stack([
                    frame[col].to_numpy(dtype=float) if col else np.full(len(frame), np.nan)
                    for col in columns
                ])
                # An oth incumbent's share is available through this generic field.
                if suffix == 'share':
                    values[incumbent == 4, 4] = frame.loc[
                        incumbent == 4, 'previous_winning_party_last_election_vote_share']
                features[f'{role}_{suffix}'] = values[rows, roles[:, number]]
        for suffix in ('share', 'projected_share'):
            features[f'{suffix}_gap'] = features[f'incumbent_{suffix}'] - features[f'contesting_party_{suffix}']
        features['previous_winning_party_last_election_vote_share'] = frame['previous_winning_party_last_election_vote_share']
        features['country/region'] = frame['country/region']
        features['national_governing_party'] = frame['incumbent']
        return features, roles


def targets(data, roles):
    winners = pd.Index(PARTIES).get_indexer(data.winner)
    if (winners < 0).any():
        raise ValueError('Winners must use con, lib, lab, natSW or oth.')
    changed = winners != roles[:, 0]
    # Destination labels are challenger positions, never the incumbent.
    destination = (roles == winners[:, None]).argmax(axis=1)
    return changed.astype(int), destination


class Stage:
    """One fitted stage, including its own numeric/categorical preprocessing."""

    def fit(self, features, labels, params):
        categorical = features.select_dtypes(include=['object', 'string']).columns.tolist()
        numeric = features.columns.difference(categorical).tolist()
        self.preprocessor = ColumnTransformer([
            ('numeric', Pipeline([
                ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                ('scale', StandardScaler()),
            ]), numeric),
            ('categorical', Pipeline([
                ('impute', SimpleImputer(strategy='constant', fill_value='__MISSING__', keep_empty_features=True)),
                ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]), categorical),
        ])
        matrix = self.preprocessor.fit_transform(features)
        self.encoder = LabelEncoder().fit(labels)
        self.classes_ = self.encoder.classes_
        self.model = None
        if len(self.classes_) > 1:
            objective = ({'objective': 'binary:logistic', 'eval_metric': 'logloss'}
                         if len(self.classes_) == 2 else
                         {'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'num_class': len(self.classes_)})
            self.model = XGBClassifier(**params, **objective, learning_rate=0.05,
                                       tree_method='hist', n_jobs=1, random_state=42)
            self.model.fit(matrix, self.encoder.transform(labels))
        return self

    def predict_proba(self, features):
        if self.model is None:
            return np.ones((len(features), 1))
        probabilities = self.model.predict_proba(self.preprocessor.transform(features)).astype(float)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, features):
        return self.classes_[self.predict_proba(features).argmax(axis=1)]


class ConditionalXGBoostModel(custom_model):
    """One candidate: P(hold) for incumbent, P(change) × P(role | change) otherwise.

    Missing previous winners use the preceding election's majority class. Missing ranking shares use
    training medians (zero for an entirely missing column, as in NN01). No
    prediction rows are filtered. Unlabelled training rows cannot supply targets.
    Retraining repeats both independent searches on the newly supplied elections.
    """

    def __init__(self):
        super().__init__('Conditional XGBoost')
        self.classes_ = np.asarray(PARTIES)
        self.mapper = None

    def train(self, data):
        training = data.dropna(subset=['winner']).reset_index(drop=True)
        if training.empty:
            raise ValueError('Training requires known winners.')
        years = pd.to_numeric(training.election, errors='raise').to_numpy()
        if not np.isfinite(years).all():
            raise ValueError('Training requires valid election years.')
        folds = []
        for year in sorted(np.unique(years))[1:]:
            earlier, later = training.loc[years < year], training.loc[years == year]
            mapper = RoleMapper().fit(earlier)
            fit_X, fit_roles = mapper.transform(earlier)
            valid_X, valid_roles = mapper.transform(later)
            folds.append((year, fit_X, targets(earlier, fit_roles),
                          valid_X, targets(later, valid_roles)))
        if not folds:
            raise ValueError('Chronological tuning requires at least two elections.')
        mapper = RoleMapper().fit(training)
        features, roles = mapper.transform(training)
        labels = targets(training, roles)
        fitted, reports, parameters = {}, {}, {}
        for stage_index, name in enumerate(('change', 'destination')):
            records = []
            for params in ParameterGrid(PARAM_GRID):
                scores = {}
                for year, fit_X, fit_y, valid_X, valid_y in folds:
                    fit_mask = np.ones(len(fit_X), dtype=bool) if stage_index == 0 else fit_y[0].astype(bool)
                    valid_mask = np.ones(len(valid_X), dtype=bool) if stage_index == 0 else valid_y[0].astype(bool)
                    if not fit_mask.any() or not valid_mask.any():
                        continue
                    stage = Stage().fit(fit_X.loc[fit_mask], fit_y[stage_index][fit_mask], params)
                    scores[year] = float(np.mean(stage.predict(valid_X.loc[valid_mask]) == valid_y[stage_index][valid_mask]))
                records.append({**params, 'mean_accuracy': np.mean(list(scores.values())) if scores else np.nan,
                                'fold_scores': scores})
            report = pd.DataFrame(records)
            # If no changed-seat fold is available, use the first grid candidate.
            best = report.mean_accuracy.idxmax() if report.mean_accuracy.notna().any() else 0
            params = {key: int(report.loc[best, key]) for key in PARAM_GRID}
            mask = np.ones(len(features), dtype=bool) if stage_index == 0 else labels[0].astype(bool)
            if not mask.any():
                raise ValueError('Destination training requires at least one changed seat.')
            fitted[name] = Stage().fit(features.loc[mask], labels[stage_index][mask], params)
            reports[name], parameters[name] = report, params
        self.mapper = mapper
        self.change_model, self.destination_model = fitted['change'], fitted['destination']
        self.search_results = reports
        self.hyperparameters = {'stages': parameters, 'learning_rate': 0.05,
                                'random_state': 42, 'tree_method': 'hist'}

    def predict_proba(self, data):
        if self.mapper is None:
            raise RuntimeError('Call train() before predict().')
        if not len(data):
            return np.empty((0, len(PARTIES)))
        features, roles = self.mapper.transform(data)
        change_probabilities = self.change_model.predict_proba(features)
        change = np.zeros(len(data))
        if 1 in self.change_model.classes_:
            change = change_probabilities[:, list(self.change_model.classes_).index(1)]
        destination = self.destination_model.predict_proba(features)
        combined = np.zeros((len(data), len(PARTIES)))
        rows = np.arange(len(data))
        for column, role in enumerate(self.destination_model.classes_):
            combined[rows, roles[:, int(role)]] = change * destination[:, column]
        combined[rows, roles[:, 0]] = 1 - change
        return combined

    def predict(self, data):
        return self.classes_[self.predict_proba(data).argmax(axis=1)]
