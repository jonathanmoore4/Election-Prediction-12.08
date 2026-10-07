"""Two conditional XGBoost stages exposed as one pipeline candidate.

Adapted from 06_xgboost_changed_extended_multilayer.ipynb. The evaluator
selects the pair of stage configurations by combined winner accuracy. Role mapping and preprocessing are
fitted on earlier rows only, including when constructing fold-specific targets.
"""
from hashlib import sha256

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import election.models.custom_model as custom_model_module
import election.models.boosting as boosting_module
import election.models.missing_data as missing_data_module

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
        self.imputer = missing_data_module.ElectionImputer().fit(data)
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
            self.model = XGBClassifier(**{**boosting_module.BOOST_DEFAULTS, **params}, **objective)
            self.model.fit(matrix, self.encoder.transform(labels))
        return self

    def predict_proba(self, features):
        if self.model is None:
            return np.ones((len(features), 1))
        probabilities = self.model.predict_proba(self.preprocessor.transform(features)).astype(float)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, features):
        return self.classes_[self.predict_proba(features).argmax(axis=1)]


class ConditionalXGBoostModel(custom_model_module.custom_model):
    """One candidate: P(hold) for incumbent, P(change) × P(role | change) otherwise.

    Missing previous winners use the preceding election's majority class. Missing ranking shares use
    training medians (zero for an entirely missing column, as in NN01). No
    prediction rows are filtered. Unlabelled training rows cannot supply targets.
    One train call fits one supplied pair of stage configurations.
    """

    def __init__(self):
        super().__init__('Conditional XGBoost')
        self.mapper = None

    def train(self, data, configuration=None, fit_context=None):
        context = fit_context or custom_model_module.fit_context()
        custom_model_module.validate_fit(data, context)
        training = data.reset_index(drop=True)
        configuration = configuration or {}
        # The cache belongs to exactly this fold, not to the model or global state.
        cache = context['cache']
        signature = sha256(pd.util.hash_pandas_object(
            training[FEATURE_COLUMNS + ['election', 'winner']], index=True).values.tobytes()).hexdigest()
        if cache.get('training_signature', signature) != signature:
            raise ValueError('Fit cache cannot be shared across different training data.')
        cache['training_signature'] = signature
        if 'role_data' not in cache:
            mapper = RoleMapper().fit(training)
            features, roles = mapper.transform(training)
            cache['role_data'] = (mapper, features, targets(training, roles))
        mapper, features, labels = cache['role_data']
        if not labels[0].any():
            raise ValueError('Conditional XGBoost requires changed-seat training rows.')
        fitted, parameters = {}, {}
        for stage_index, name in enumerate(('change', 'challenger')):
            params = {**boosting_module.BOOST_DEFAULTS, 'n_estimators': 25, 'max_depth': 2,
                      **{key: value for key, value in configuration.items() if '.' not in key}}
            params.update({key.split('.', 1)[1]: value for key, value in configuration.items()
                           if key.startswith(name + '.')})
            key = (name, tuple(sorted(params.items())))
            if key not in cache:
                mask = (np.ones(len(features), dtype=bool) if stage_index == 0
                        else labels[0].astype(bool))
                cache[key] = Stage().fit(features.loc[mask], labels[stage_index][mask], params)
            fitted[name] = cache[key]
            parameters.update({f'{name}.{key}': value for key, value in params.items()})
        self.mapper = mapper
        self.change_model = fitted['change']
        self.destination_model = fitted['challenger']
        return self._record(data, parameters, context)

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
        return self._align_probabilities(combined, PARTIES)

    def predict(self, data):
        return self.classes_[self.predict_proba(data).argmax(axis=1)]


conditionalxgboosthyperparameters = {f'{stage}.{key}': list(values) for stage in ('change', 'challenger') for key, values in PARAM_GRID.items()}
TRAINING_METADATA = {
    'model_id': 'conditional_xgboost',
    'architecture_id': 'conditional_xgboost_model-v1',
    'training_protocol': 'full-history-v1',
    'fixed_settings': dict(boosting_module.BOOST_DEFAULTS),
    'features': list(FEATURE_COLUMNS),
    'supported_hyperparameters': sorted(set(boosting_module.BOOST_DEFAULTS) | set(conditionalxgboosthyperparameters)),
}


def ConditionalXGBoost(train_data, test_data, hyperparameters, *, fit_records=None,
           cache=None, return_details=False, output_dir=None, return_model=False):
    """Fit one configuration and score known winners; no MLflow operations."""
    from election.models.model_function import evaluate_fit
    return evaluate_fit(ConditionalXGBoostModel, train_data, test_data, hyperparameters,
                        metadata=TRAINING_METADATA,
                        context=custom_model_module.fit_context(cache=cache),
                        return_details=return_details, output_dir=output_dir, return_model=return_model)
