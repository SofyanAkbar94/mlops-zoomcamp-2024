#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[2]:


get_ipython().system('python -V')


# In[3]:


import pickle
import pandas as pd


# In[4]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[7]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[8]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')


# In[9]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[10]:


import numpy as np

std_dev_pred = np.std(y_pred)
print("Standard Deviation of Predicted Duration:", std_dev_pred)


# In[14]:


# Define year and month
year = 2023
month = 3

# Create an artificial ride_id column
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[15]:


# Create a results dataframe
df_result = pd.DataFrame({
    'ride_id': df['ride_id'],
    'predicted_duration': y_pred
})

# Define the output file path
output_file = 'predictions.parquet'

# Save the results to a Parquet file
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

# Check the size of the output file
import os
file_size = os.path.getsize(output_file)
print(f"Size of the output file: {file_size/1024/1024} MB")


# In[ ]:




