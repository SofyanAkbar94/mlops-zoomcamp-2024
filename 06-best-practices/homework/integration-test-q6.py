import os
import pandas as pd
import s3fs

# Run the batch.py script for January 2023
os.system('python batch_q6.py 2023 1')

# Read the result from S3
output_file = 's3://nyc-duration/output/yellow_tripdata_2023-01.parquet'
options = {'client_kwargs': {'endpoint_url': 'http://localhost:4566'}}

df_result = pd.read_parquet(output_file, storage_options=options)

# Calculate the sum of predicted durations
sum_predicted_durations = df_result['predicted_duration'].sum()

print(f'Sum of predicted durations: {sum_predicted_durations}')
