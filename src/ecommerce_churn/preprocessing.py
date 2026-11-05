from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class CatBoostPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_columns, categorical_columns):
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns

    def fit(self, X, y=None):
        self.numerical_medians_ = X[self.numerical_columns].median()
        return self

    def transform(self, X):
        transformed = X[self.numerical_columns + self.categorical_columns].copy()
        transformed[self.numerical_columns] = transformed[self.numerical_columns].fillna(self.numerical_medians_)
        for column in self.categorical_columns:
            transformed[column] = transformed[column].astype("string").fillna("__MISSING__").astype(str)
        return transformed


def make_preprocessor(numerical_columns, categorical_columns, scale_numeric=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numerical_columns),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_columns),
    ])
