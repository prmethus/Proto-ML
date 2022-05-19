# Creating the input pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from feature_selection import *
from settings import *


def create_input_pipeline(X, y, mode):
    columns_by_dtype = filter_columns_by_score(X, y, mode)
    col_transformers = []

    if len(columns_by_dtype["numerical"]) > 0:
        Numerical_Transformer = Pipeline(
            steps=[("Numerical Imputer", SimpleImputer(strategy="mean"))]
        )
        col_transformers.append(
            (
                "Numerical_Transformer",
                Numerical_Transformer,
                columns_by_dtype["numerical"],
            )
        )

    if len(columns_by_dtype["ordinal_encoding"]) > 0:
        Ordinal_Transformer = Pipeline(
            steps=[
                ("Ordinal Imputer", SimpleImputer(strategy="most_frequent")),
                ("Ordinal Encoder", OrdinalEncoder()),
            ]
        )

        col_transformers.append(
            (
                "Ordinal_Transformer",
                Ordinal_Transformer,
                columns_by_dtype["ordinal_encoding"],
            )
        )

    if len(columns_by_dtype["one_hot_encoding"]) > 0:
        One_Hot_Transformer = Pipeline(
            steps=[
                ("One Hot Imputer", SimpleImputer(strategy="most_frequent")),
                ("One Hot Encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        col_transformers.append(
            (
                "One Hot Transformer",
                One_Hot_Transformer,
                columns_by_dtype["one_hot_encoding"],
            )
        )

    Input_Pipeline = ColumnTransformer(transformers=col_transformers)

    Input_Pipeline.fit(X, y)
    return Input_Pipeline
