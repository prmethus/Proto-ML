# Creating the input pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
from feature_selection import *

def create_input_pipeline(input_data, y, mode):
    columns_by_dtype = filter_columns_by_score(input_data, y, mode)
    
    numerical_features = columns_by_dtype["numerical"]
    one_hot_encoding_features = columns_by_dtype["one_hot_encoding"]
    ordinal_encoding_features = columns_by_dtype["ordinal_encoding"]

    if len(numerical_features) > 1:
        