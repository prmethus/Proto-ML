# from input_pipeline import create_input_pipeline
# from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
import stringcolor


class ML_Pipeline:
    @classmethod
    def assertion_test(task:str):
        try:
            assert(task == "classification" or task == "regression")
        except AssertionError:
            raise Exception(stringcolor.cs("Wrong value set to VARIABLE: task. It should be 'regression' or 'classification'.","red").bold())

    
    def train():


        param_grid = 
        search = GridSearchCV(model, )
        
        

    train("Task", 2)

    def save_model