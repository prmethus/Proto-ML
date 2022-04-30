from input_pipeline import create_input_pipeline
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
import os, stringcolor, joblib
from sklearn.metrics import r2_score


class ML_Pipeline:

    def __init__(self, mode):
        self.mode = mode
        self.assertion_test()

    def assertion_test(self ):
        try:
            assert((self.mode == "classification") or (self.mode == "regression"))      
        except AssertionError:
            raise Exception(stringcolor.cs("Wrong value set to VARIABLE: task. It should be 'regression' or 'classification'.","red").bold())

    
    def fit(self, X, y):
        
        print(stringcolor.cs("Creating Input pipeline...", "cyan"))
        self.input_pipeline = create_input_pipeline(X, y, self.mode)

        Transformed_Input_Data = self.input_pipeline.transform(X)

        parameters_grid = {'n_estimators':[100,200,300,400,500], 'booster':['gbtree', 'dart']}

        if self.mode == "classification":
            print(stringcolor.cs("Training XGBClassifier model...", "cyan"))
            model = XGBClassifier(random_state=0, verbosity=0)

        elif self.mode == "regression":
            print(stringcolor.cs("Training XGBRegressor model...", "cyan"))
            model = XGBRegressor(random_state=0, verbosity=0)

        search = GridSearchCV(model, param_grid=parameters_grid)
        search.fit(Transformed_Input_Data, y)
        self.model = search.best_estimator_
        print(stringcolor.cs("Model Trained. Accuracy: %5.3f" % self.model.score(Transformed_Input_Data, y), "cyan").bold())

    def predict(self, X):
        Transformed_Data = self.input_pipeline.transform(X)
        output = self.model.predict(Transformed_Data)
        return output

    def score(self, X, y):
        Transformed_Data = self.input_pipeline.transform(X)
        output = self.model.predict(Transformed_Data)
        score = r2_score(y, output)
        return score

    def save(self, directory):
        if not os.path.exists(directory):
            try:
                os.mkdir(directory)
            except Exception as e:
                print(e)
                raise Exception(stringcolor.cs("PATH ERROR! There's probably something wrong with the directory you specified!", "red").bold())
                
        model_path = os.path.join(directory, "model.pkl")
        input_pipeline_path = os.path.join(directory, "input_pipeline.pkl")
        joblib.dump(self.model, model_path)
        joblib.dump(self.input_pipeline, input_pipeline_path)
