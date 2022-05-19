from protoml.feature_selection import filter_columns_by_score
from protoml.input_pipeline import create_input_pipeline
from protoml.data_split import split_data
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
import os, stringcolor, joblib
from sklearn.metrics import r2_score
from protoml import settings
import joblib


class base:
    @classmethod
    def save(self, ml_pipeline, directory):
        if not os.path.exists(directory):
            try:
                os.mkdir(directory)
            except Exception as e:
                print(e)
                raise Exception(
                    stringcolor.cs(
                        "PATH ERROR! There's probably something wrong with the directory you specified!",
                        "red",
                    ).bold()
                )

        model_path = os.path.join(directory, "ml_pipeline.pkl")

        joblib.dump(ml_pipeline, model_path)

        with open(os.path.join(directory, "input_features.txt"), "w") as i_f:
            i_f.write("/^/".join(settings.inputFeatures))


    @classmethod
    def load(self, directory):
        ml_pipeline_path = os.path.join(directory, "ml_pipeline.pkl")
        ml_pipeline = joblib.load(ml_pipeline_path)
        
        with open(os.path.join(directory, "input_features.txt"), "r") as i_f:
            inp_features = i_f.read().split("/^/")
        ml_pipeline.get_features(inp_features)

        return ml_pipeline



class ML_Pipeline:
    def __init__(self, mode):
        
        self.mode = mode

        try:
            assert (self.mode == "classification") or (self.mode == "regression")
        except AssertionError:
            raise Exception(
                stringcolor.cs(
                    "Wrong value set to VARIABLE: task. It should be 'regression' or 'classification'.",
                    "red",
                ).bold()
            )

    def fit(self, X, y):

        print(stringcolor.cs("Creating Train and Test splits...", "cyan"))
        
        X_train, y_train, X_validation, y_validation = split_data(X, y, self.mode)
        
        print(stringcolor.cs("Creating Input pipeline...", "cyan"))

        self.input_pipeline = create_input_pipeline(X_train, y_train, self.mode)

        Transformed_Input_Data = self.input_pipeline.transform(X_train)

        Transformed_Validation_Data = self.input_pipeline.transform(X_validation)

        parameters_grid = {
            "n_estimators": [100, 200, 300, 400, 500],
            "booster": ["gbtree", "dart"],
        }

        if self.mode == "classification":
            print(stringcolor.cs("Training XGBClassifier model...", "cyan"))
            model = XGBClassifier(random_state=0, verbosity=0)

        elif self.mode == "regression":
            print(stringcolor.cs("Training XGBRegressor model...", "cyan"))
            model = XGBRegressor(random_state=0, verbosity=0)

        search = GridSearchCV(model, param_grid=parameters_grid)
        search.fit(Transformed_Input_Data, y_train)
        self.model = search.best_estimator_

        print(
            stringcolor.cs(
                "Train Accuracy: {} | Test Accuracy: {}".format(
                    self.model.score(Transformed_Input_Data, y_train),
                    self.model.score(Transformed_Validation_Data, y_validation),
                ),
                "purple",
            ).bold()
        )

    def predict(self, X):

        Transformed_Data = self.input_pipeline.transform(X)
        output = self.model.predict(Transformed_Data)
        return output

    def score(self, X, y):
        Transformed_Data = self.input_pipeline.transform(X)
        output = self.model.predict(Transformed_Data)
        score = r2_score(y, output)
        return score

    def get_features(self, inp_features):
        settings.inputFeatures = inp_features