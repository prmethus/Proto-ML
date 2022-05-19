# Incomplete

import pandas as pd

def clean_data(df):

    qualitative_data = df.select_dtypes("object")
    qualitative_data = clean_string_data(qualitative_data)


def clean_string_data(qualitative_data):
    qualitative_columns = qualitative_data.columns.tolist()
    qualitative_data[qualitative_columns] = qualitative_data[qualitative_columns].apply(
        lambda column: column.map(lambda x: x.strip().lower())
    )
