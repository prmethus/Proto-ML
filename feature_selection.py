from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import pandas as pd

def calculate_mi_score(input_data, y, mode):
    mi_input_data = input_data.copy()
    
    for col in mi_input_data.select_dtypes(["object"]):
        mi_input_data[col], _ = mi_input_data[col].factorize()
    if mode == "classification":
        mi_score = mutual_info_classif(mi_input_data, y)
    elif mode == "regression":
        mi_score = mutual_info_regression(mi_input_data, y)

    ndf = pd.Series(mi_score, index=mi_input_data.columns).sort_values(ascending=False)

    return ndf 

def filter_columns_by_score(input_data, y, mode, mi_score_threshold = 0.5, one_hot_encoder_nlimit=10):

    input_data_copy = input_data.copy()
    columns_by_dtype = {"numerical":[],"one_hot_encoding":[], "ordinal_encoding":[]}
    mi_score = calculate_mi_score(input_data, y, mode)
    threshold_surpassing_cols = mi_score.loc[mi_score > mi_score_threshold].index.tolist()

    numerical_columns = input_data[threshold_surpassing_cols].select_dtypes(["int", "float"]).columns.tolist()
    categorical_columns = input_data[threshold_surpassing_cols].select_dtypes(["object","category"]).columns.tolist()

    columns_by_dtype["numerical"] = numerical_columns

    for categorical_column in categorical_columns:
        if input_data_copy[categorical_column].nunique() <= one_hot_encoder_nlimit:
            columns_by_dtype["one_hot_encoding"].append(categorical_column)
        else:
            columns_by_dtype["ordinal_encoding"].append(categorical_column)

    return columns_by_dtype