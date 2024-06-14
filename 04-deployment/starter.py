import pickle
import pandas as pd
import argparse
import numpy as np

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def main(year, month):
    file_path = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    
    # Load and process the data
    df = read_data(file_path)
    
    # Convert categorical columns to dictionaries and transform
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    
    # Predict the durations
    y_pred = model.predict(X_val)
    
    # Create an artificial ride_id column
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    # Create a results dataframe
    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })

    # Calculate and print the mean predicted duration
    mean_pred_duration = np.mean(y_pred)
    print(f"Mean predicted duration: {mean_pred_duration}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data')
    parser.add_argument('--month', type=int, required=True, help='Month of the data')

    args = parser.parse_args()
    main(args.year, args.month)