#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
from datetime import datetime
import s3fs

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def read_data(filename, categorical):
    df = pd.read_parquet(filename, storage_options={'client_kwargs': {'endpoint_url': 'http://localhost:4566'}})
    return df

def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def save_data(df, filename, options):
    df.to_parquet(
        filename,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

def main(year, month):
    input_file = f's3://nyc-duration/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f's3://nyc-duration/output/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']
    df = read_data(input_file, categorical)
    df = prepare_data(df, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    save_data(df_result, output_file, {'client_kwargs': {'endpoint_url': 'http://localhost:4566'}})

if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)
