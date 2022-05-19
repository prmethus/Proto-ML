# Creating the input pipeline
from itertools import chain
import stringcolor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from protoml import settings

class feature_selector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):      
        new_X = X.loc[:,settings.inputFeatures]
        return new_X

from sklearn.base import BaseEstimator, TransformerMixin
from protoml.feature_selection import filter_columns_by_score


def create_input_pipeline(X, y, mode):
    columns_by_dtype = filter_columns_by_score(X, y, mode)
    settings.inputFeatures = list(chain.from_iterable(columns_by_dtype.values()))  
    col_transformers = []
    
    print(stringcolor.cs(f"Selected Input Features ({len(settings.inputFeatures)}): {', '.join(settings.inputFeatures)}", "green").bold())


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

    Columns_Transformer = ColumnTransformer(transformers=col_transformers)

    Input_Pipeline = Pipeline(steps=[
        ("Feature Selection", feature_selector()),
        ("Columns Transformer", Columns_Transformer)
    ])

    Input_Pipeline.fit(X, y)
    return Input_Pipeline