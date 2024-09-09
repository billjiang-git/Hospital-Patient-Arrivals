import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw



############ MSE #############

def compute_mse_between_dataframes(df1, df2, column_name='arrival_date'):
    # Ensure the arrival_date column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df1[column_name]):
        df1[column_name] = pd.to_datetime(df1[column_name])
    if not pd.api.types.is_datetime64_any_dtype(df2[column_name]):
        df2[column_name] = pd.to_datetime(df2[column_name])
    # Convert datetime to timestamps (seconds since epoch)
    df1_timestamps = (df1[column_name] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    df2_timestamps = (df2[column_name] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    # Calculate and return MSE
    mse = mean_squared_error(df1_timestamps, df2_timestamps)
    return mse




############ EMD #############

def compute_emd_between_dataframes(df1, df2, column_name='arrival_date'):
    # Extract the column data from each dataframe
    if not pd.api.types.is_datetime64_any_dtype(df1[column_name]):
        df1[column_name] = pd.to_datetime(df1[column_name])
    if not pd.api.types.is_datetime64_any_dtype(df2[column_name]):
        df2[column_name] = pd.to_datetime(df2[column_name])
    # Convert datetime to timestamps (seconds since epoch)
    df1_timestamps = (df1[column_name] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    df2_timestamps = (df2[column_name] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    emd = wasserstein_distance(df1_timestamps, df2_timestamps)
    
    return emd





############ DTW #############

def compute_dtw_between_dataframes(df1, df2, column_name='arrival_date'):
    # Helper function to convert datetime strings or datetime objects to seconds since start of the day
    def to_seconds(datetime_series):
        return np.array([[(pd.to_timedelta(time).total_seconds() if isinstance(time, str) else (time - time.normalize()).total_seconds())] 
                         for time in datetime_series])
    # Convert 'arrival_date' columns from both dataframes to seconds since start of the day and ensure it's in list format
    sequence1 = to_seconds(df1[column_name])
    sequence2 = to_seconds(df2[column_name])
    # Compute the Dynamic Time Warping distance
    distance, path = fastdtw(sequence1, sequence2, dist=euclidean)
    return distance
