from typing import Tuple

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from mlops.utils.data_preparation.cleaning import clean
from mlops.utils.data_preparation.feature_engineering import combine_features
from mlops.utils.data_preparation.feature_selector import select_features
from mlops.utils.data_preparation.splitters import split_on_value

import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def transform(df: pd.DataFrame, **kwargs) -> Tuple[DictVectorizer, LinearRegression]:
    data_dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    dv = DictVectorizer()
    X = dv.fit_transform(data_dicts)

    y = df['duration'].values

    model = LinearRegression()
    model.fit(X, y)

    print("Intercept:", model.intercept_)
    
    return dv, model