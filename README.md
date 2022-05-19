# protoml (UNDER DEVELOPMENT)
## Create Prototype ML Models ASAP!
Currently Supported Modes = 'classification', 'regression'

### Steps:
1. Feature Selection with MI Score (Default Threshold: 0.5)
2. Encoding for Categorical Data
3. Pipeline creation with Scikit-Learn
4. Model training with XGBoost

### Installation:
Run `python3 setup.py install` to install protoml


### Usage:
protoml.ML_pipeline(mode)
- Creates the ML pipeline
.fit(X,y) For Splitting the data and training the model on it
.predict(X) For making predictions
.score(X,y) Shows the R2 score

### Tools:
protoml.base - Utilities for working with protoml
protoml.visualization - Uses Matplotlib and Seaborn backend for


### protoml.base
protoml.base.save(ml_pipeline, directory)
- Saves the trained pipeline in the specified directory

protoml.base.load(directory)
- Loads the trained pipeline from the specified directory


#### THIS IS A BETA VERSION
FURTHER IMPROVEMENTS REQUIRED: Early Stopping to reduce Overfitting, Automated Feature Engineering, Command Line Utility, Automated Deployment with FastAPI and Docker
