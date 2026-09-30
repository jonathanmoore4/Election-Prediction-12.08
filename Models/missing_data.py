"""Training-fitted fills applied after SQL; election is metadata, not a predictor."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ElectionImputer(TransformerMixin, BaseEstimator):
    """Fill previous winners using the preceding available election's seat mode.

    For the earliest training election, use its known previous-winner mode.
    Labels passed to transform are never inspected. Numeric medians are fitted
    on training rows only; entirely missing numeric columns use zero.
    """
    def fit(self, X, y=None):
        years = pd.to_numeric(X['election'])
        labels = X['winner'] if 'winner' in X else y
        self.winner_modes_ = {}
        if labels is not None:
            labels = pd.Series(np.asarray(labels), index=X.index)
            for year in sorted(years.unique()):
                mode = labels.loc[years == year].dropna().mode()
                if len(mode):
                    self.winner_modes_[year] = mode.iloc[0]
        self.previous_modes_ = {}
        for year in sorted(years.unique()):
            mode = X.loc[years == year, 'previous_winner'].dropna().mode()
            if len(mode):
                self.previous_modes_[year] = mode.iloc[0]
        mode = X.previous_winner.dropna().mode()
        if not len(mode) and not self.winner_modes_:
            raise ValueError('Previous-winner imputation requires historical party labels.')
        self.fallback_ = mode.iloc[0] if len(mode) else next(iter(self.winner_modes_.values()))
        self.medians_ = {}
        for column in X.select_dtypes(include='number').columns.difference(['election', 'winner', 'previous_winner', 'country/region', 'incumbent']):
            values = X[column].dropna()
            self.medians_[column] = values.median() if len(values) else 0.0
        return self

    def transform(self, X):
        result = X.copy()
        result["previous_winner"] = result.previous_winner.astype(object)
        years = pd.to_numeric(result.election)
        for year in years.loc[result.previous_winner.isna()].unique():
            earlier = [previous for previous in self.winner_modes_ if previous < year]
            value = (self.winner_modes_[max(earlier)] if earlier else
                     self.previous_modes_.get(year, self.fallback_))
            result.loc[(years == year) & result.previous_winner.isna(), 'previous_winner'] = value
        for column, median in self.medians_.items():
            if column in result:
                result[column] = result[column].fillna(median)
        return result
