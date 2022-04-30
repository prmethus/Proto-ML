import pandas as pd
from feature_selection import *

data = pd.read_csv("/home/sarzil/Programming/Projects/AutoML/FuelConsumption.csv")

X = data.drop(columns=["CO2EMISSIONS"])
y = data["CO2EMISSIONS"]
score = calculate_mi_score(X, y, mode="regression")
print(score.loc[score > 0.5].index.tolist())

print(filter_columns_by_score(X, y, mode="regression"))