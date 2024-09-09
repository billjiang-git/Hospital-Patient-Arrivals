# General data processing functions



# calculates time between arrivals, takes in the dataframe
def interarrival_times(data):
    data['interarrival_time'] = data['arrival_date'].diff().dt.total_seconds()
    return data



# adjust two dataframes to be the same length for MSE calculation
def adjust_datasets_to_shortest_length(list_of_dataframes):
    # Determine the minimum length between the two dataframes
    min_length = min(len(df) for df in list_of_dataframes)
    # Slice both dataframes to the minimum length
    adjusted_dataframes = [df.iloc[:min_length] for df in list_of_dataframes]
    return adjusted_dataframes