import os
import pickle
import numpy as np
from hyperopt import hp, fmin, tpe, Trials
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def objective(params):
    # Extract hyperparameters
    max_depth = params['max_depth']
    n_estimators = params['n_estimators']

    # Load data
    X_train, y_train = load_pickle("./output/train.pkl")
    X_val, y_val = load_pickle("./output/val.pkl")

    # Train model
    rf = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=0)
    rf.fit(X_train, y_train)

    # Predict on validation set
    y_pred = rf.predict(X_val)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    # Log hyperparameters and RMSE
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)

    return rmse


# Define hyperparameter search space
space = {
    'max_depth': hp.choice('max_depth', range(5, 15)),
    'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400, 500])
}

# Initialize Trials object
trials = Trials()

# Run hyperparameter optimization
best = fmin(objective, space, algo=tpe.suggest, max_evals=10, trials=trials)

# Get best hyperparameters
best_params = space_eval(space, best)

print("Best hyperparameters:", best_params)
