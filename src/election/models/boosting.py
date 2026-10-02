"""One-fit boosted classifier with explicit defaults and party-label encoding."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier
import election.models.missing_data as missing_data_module

BOOST_DEFAULTS = dict(n_estimators=100, max_depth=3, learning_rate=0.05,
                      min_child_weight=1, gamma=0, subsample=1, colsample_bytree=1,
                      reg_alpha=0, reg_lambda=1, tree_method='hist', random_state=42,
                      n_jobs=1)


class LabelledBoosting(ClassifierMixin, BaseEstimator):
    def __init__(self, parameters=None):
        self.parameters = parameters

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.model_ = None
        if len(self.classes_) > 1:
            self.model_ = XGBClassifier(**(self.parameters or BOOST_DEFAULTS),
                                       objective='multi:softprob',
                                       num_class=len(self.classes_), eval_metric='mlogloss')
            self.model_.fit(X, self.encoder_.transform(y))
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        if self.model_ is None:
            return np.ones((len(X), 1))
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def fit_boosting_pipeline(data, feature_columns, categorical, parameters):
    numeric = [column for column in feature_columns if column not in categorical]
    preprocessing = ColumnTransformer([
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='__MISSING__',
                                    keep_empty_features=True)),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categorical),
        ('numeric', SimpleImputer(strategy='median', keep_empty_features=True), numeric),
    ])
    pipeline = Pipeline([
        ('missing_data', missing_data_module.ElectionImputer()), ('preprocessing', preprocessing),
        ('classifier', LabelledBoosting(parameters)),
    ])
    training = data.dropna(subset=['winner'])
    pipeline.fit(training[feature_columns + ['election']], training.winner)
    return pipeline
