import pandas as pd
from model import *
from input_pipeline import *
data = pd.read_csv("/home/sarzil/Programming/Projects/AutoML/FuelConsumption.csv")

X = data.drop(columns=["CO2EMISSIONS"])
y = data["CO2EMISSIONS"]

pipe = ML_Pipeline(mode="regression")
pipe.fit(X,y)
pipe.save("/home/sarzil/Programming/Projects/AutoML/trained_model")
# score = calculate_mi_score(X, y, mode="regression")
# print(score.loc[score > 0.5].index.tolist())

# print(filter_columns_by_score(X, y, mode="regression"))

# input_pipeline = create_input_pipeline(X, y, mode='regression')
# transformed = input_pipeline.fit_transform(X,y)
# print(transformed)