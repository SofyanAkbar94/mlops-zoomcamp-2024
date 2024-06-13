import mlflow
import pickle
import mlflow.sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from typing import Tuple

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(output: Tuple[DictVectorizer, LinearRegression], **kwargs) -> None:
    dv, model = output

    mlflow.set_tracking_uri('http://mlflow:5001')
    mlflow.set_experiment("homework")
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        mlflow.sklearn.log_model(model, 'linear_regression_model')
        
        artifact_path = 'dict_vectorizer.pkl'
        with open(artifact_path, 'wb') as f:
            pickle.dump(dv, f)
        mlflow.log_artifact(artifact_path, artifact_path)
        import os
        model_pickle_path = 'linear_regression_model.pkl'
        with open(model_pickle_path, 'wb') as f:
            pickle.dump(model, f)
        model_size_bytes = os.path.getsize(model_pickle_path)
        
        mlflow.log_metric('model_size_bytes', model_size_bytes)
        
