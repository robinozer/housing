from sklearn.base import BaseEstimator, TransformerMixin

class ConvertToInt64(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.columns:
            X[col] = X[col].astype('Int64')
        return X
        