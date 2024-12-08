# transformers.py
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer

class Discretizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_indices, n_bins=100):
        self.feature_indices = feature_indices
        self.discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    
    def fit(self, X, y=None):
        for idx in self.feature_indices:
            self.discretizer.fit(X[:, idx].reshape(-1, 1))
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for idx in self.feature_indices:
            X_transformed[:, idx] = self.discretizer.transform(X[:, idx].reshape(-1, 1)).flatten()
        return X_transformed