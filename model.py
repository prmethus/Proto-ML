# from input_pipeline import create_input_pipeline
# from xgboost import XGBClassifier, XGBRegressor
import stringcolor

def train(X, y, task:str):
    try:
        assert(task == "classification" or task == "regression")
    except AssertionError:
        raise Exception(stringcolor.cs("Wrong value set to VARIABLE: task. It should be 'regression' or 'classification'.","red").bold())

    
    

train("Task", 2)