import pandas as pd
import numpy as np
import re
from datetime import timedelta, datetime
import ruptures as rpt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from plotting_arrivals import (plot_combined_with_clusters, plot_combined_colored_intervals)







def change_point_on_dataset(data, interval_hours, num_k, bkps):

    data['arrival_date'] = pd.to_datetime(data['arrival_date'])
    hourly_counts = data.resample(f'{interval_hours}h', on='arrival_date').size()

    algo = rpt.Binseg(model="l2").fit(hourly_counts.values)
    n = len(hourly_counts)
    sigma = np.std(hourly_counts.values)
    # penalty = np.log(np.log(np.log(n))) * sigma**2  # Adjust penalty as needed
    # penalty = 1
    # result = algo.predict(pen=penalty)
    result = algo.predict(n_bkps=bkps)
    
    # Plotting the arrivals data using plt directly
    
    # plt.figure(figsize=(14, 7))
    # plt.plot(hourly_counts.index, hourly_counts, linestyle='-', color='blue', label='Hourly Arrivals')

    # # Annotating the detected change points directly with plt.axvline
    # for bk in result[:-1]:  # Exclude the last point as it is just the end of the data
    #     if bk < len(hourly_counts):  # Ensure the breakpoint index is within the range of the plot
    #         plt.axvline(x=hourly_counts.index[bk], color='red', linestyle='--', label='Change Point' if bk == result[0] else "")

    # ############## Setting x-axis ticks and formatting
    # plt.xticks(hourly_counts.index, rotation=45)
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator())  # Major ticks per day
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # plt.title(f'Arrivals Every {interval_hours} Hours and Change Points Over Dataset')
    # plt.xlabel('Date')
    # plt.ylabel('Number of Arrivals')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    
    def create_time_buckets():
        # Ensure 'arrival_date' is in datetime format
        data['arrival_date'] = pd.to_datetime(data['arrival_date'])

        # Initialize the dictionary to hold the data buckets
        time_buckets = {}

        # Get the start and end of the dataset
        start_date = data['arrival_date'].min().floor('h')  # Round down to the nearest hour
        end_date = data['arrival_date'].max().ceil('h')  # Round up to the nearest hour

        # Generate buckets from start_date to end_date with step of 'interval_hours'
        current_time = start_date
        while current_time < end_date:
            # Define the end of the current bucket
            next_time = current_time + pd.Timedelta(hours=interval_hours)

            # Create the bucket key in the format "YYYY-MM-DD HH:MM-HH:MM"
            bucket_key = f"{current_time.strftime('%Y-%m-%d %H:%M')}-{next_time.strftime('%H:%M')}"

            # Filter data within the current time bucket
            mask = (data['arrival_date'] >= current_time) & (data['arrival_date'] < next_time)
            filtered_data = data.loc[mask]

            # Store the bucket if there is data
            if not filtered_data.empty:
                time_buckets[bucket_key] = filtered_data[['arrival_date', 'interarrival_time']]

            # Move to the next time bucket
            current_time = next_time
        
        # print(time_buckets)

        return time_buckets

    def fully_dynamic_buckets(L_breakpoints, arrival_data):
        times = list(arrival_data.keys())
        dynamic_buckets = {}
        start_index = 0
        stats = []

        for i, end_index in enumerate(L_breakpoints + [len(times)]):  # Ensure last bucket extends to include all data
            if start_index < len(times) and end_index <= len(times):
                start_time = times[start_index]
                if end_index == len(times):  # Handling the last bucket specifically
                    end_time = times[-1]
                else:
                    end_time = times[end_index - 1]

                # Change here: Include the entire datetime string for start and end times
                start_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2})-\d{2}:\d{2}', start_time)
                if start_match:
                    start_time_formatted = start_match.group(1)  # Extracts 'YYYY-MM-DD HH:MM' from the start time string

                # Extract and conditionally adjust the end time
                end_match = re.match(r'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})-(\d{2}:\d{2})', end_time)
                if end_match:
                    date_part = end_match.group(1)  # Extract the date part '2012-11-22'
                    end_time = end_match.group(3)  # Extract the end time '00:00' or other times

                    # Convert date_part to a datetime object
                    date_obj = datetime.strptime(date_part, '%Y-%m-%d')

                    if end_time == '00:00':
                        # Since time is '00:00', increment the date to the next day
                        next_day = date_obj + timedelta(days=1)
                        result_str = next_day.strftime('%Y-%m-%d') + ' ' + end_time  # 'YYYY-MM-DD 00:00'
                    else:
                        # Regular time, return same day with end time
                        result_str = date_part + ' ' + end_time

                # Create the bucket key
                bucket_key = f"{start_time_formatted} to {result_str}"
                bucket_frames = [arrival_data[time] for time in times[start_index:end_index]]
                # print(bucket_frames)
                dynamic_buckets[bucket_key] = pd.concat(bucket_frames)
                start_index = end_index

        for key, df in dynamic_buckets.items():
            if not df.empty and 'interarrival_time' in df.columns:
                df['Mean_Interarrival'] = df['interarrival_time'].mean()
                df['Variance_Interarrival'] = df['interarrival_time'].var()
                df['Skewness_Interarrival'] = df['interarrival_time'].skew()

        # print(dynamic_buckets)
        return dynamic_buckets

    
    def plot_moments_with_clusters(buckets, num_k):
        # Prepare the data for clustering
        # print(buckets.values())
        moments_data = pd.DataFrame({
            'Mean_Interarrival': [df['Mean_Interarrival'].iloc[0] if 'Mean_Interarrival' in df.columns else None for df in buckets.values()],
            'Variance_Interarrival': [df['Variance_Interarrival'].iloc[0] if 'Variance_Interarrival' in df.columns else None for df in buckets.values()],
            'Skewness_Interarrival': [df['Skewness_Interarrival'].iloc[0] if 'Skewness_Interarrival' in df.columns else None for df in buckets.values()]
        }).dropna()
        # print(moments_data)

        if len(moments_data) < num_k:

            # print(f"Reduce number of clusters from {num_k} to less than or equal to {len(moments_data)} due to insufficient data points.")
            
            return {}, {}

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=num_k, random_state=0, n_init=10)
        moments_data['cluster_label'] = kmeans.fit_predict(moments_data)
        # print(moments_data)

        grouped_data = moments_data.groupby('cluster_label').mean()
        # print(grouped_data)

        # Map the dataframes to their new clusters
        bucket_keys = list(buckets.keys())
        cluster_dict = {}
        cluster_intervals = {}

        
        # plt.figure(figsize=(14, 7))
        # color_map = cm.get_cmap('tab10', num_k)
        # ax = plt.gca()
        # color_map = plt.get_cmap('viridis', num_k)
        

        for index, row in moments_data.iterrows():
            bucket_key = bucket_keys[index]
            label = row['cluster_label']
            mean = grouped_data.loc[label, 'Mean_Interarrival']
            variance = grouped_data.loc[label, 'Variance_Interarrival']
            skewness = grouped_data.loc[label, 'Skewness_Interarrival']
            key = f"cluster number: {label}, mean: {mean}, variance: {variance}, skewness: {skewness}"

            # if key in cluster_dict:
            #     cluster_dict[key].append(buckets[bucket_key])
            # else:
            #     cluster_dict[key] = [buckets[bucket_key]]

            current_df = buckets[bucket_key][['arrival_date', 'interarrival_time']]

            start_date = current_df['arrival_date'].min()
            end_date = current_df['arrival_date'].max()

            if key in cluster_dict:
                cluster_dict[key] = pd.concat([cluster_dict[key], current_df])
            else:
                cluster_dict[key] = current_df



        cluster_intervals = {i: [] for i in range(num_k)}
        for index, row in moments_data.iterrows():
            bucket_key = list(buckets.keys())[index]
            label = row['cluster_label']
            current_df = buckets[bucket_key]

            start_date = current_df['arrival_date'].min()
            end_date = current_df['arrival_date'].max()
            cluster_intervals[label].append((start_date, end_date))
        
        # print(cluster_intervals)

        # Plot intervals
        
        # for label, intervals in cluster_intervals.items():
        #     # print(intervals)
        #     for start_date, end_date in intervals:
        #         plt.plot([start_date, end_date], [label]*2, color=color_map(label), marker='|', markersize=10, label=f'Cluster {label}' if start_date == intervals[0][0] else "")
        
        # plt.xlabel('Date')
        # plt.ylabel('Cluster Number')
        # plt.title('Clustered Time Intervals Over Dataset')
        # plt.legend(title='Cluster Labels')
        # plt.grid(True)
        # plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.show()

        # print(cluster_dict)
        # for df in cluster_dict.values():
        #     print(len(df))
        # print(sum(len(df) for df in cluster_dict.values()))



        # # 2D plot with first and second moment
        # plt.figure(figsize=(10, 6))
        # plt.scatter(moments_data['Mean_Interarrival'], moments_data['Variance_Interarrival'], c=moments_data['cluster_label'], cmap='viridis', marker='o')
        # plt.title('Cluster Visualization')
        # plt.xlabel('Mean')
        # plt.ylabel('Variance')
        # plt.colorbar(label='Cluster Label')
        # plt.show()


        # # Create 3D plot
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
        # scatter = ax.scatter(moments_data['Mean_Interarrival'], moments_data['Variance_Interarrival'], moments_data['Skewness_Interarrival'], 
        #                     c=moments_data['cluster_label'], cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.6)

        # # Labeling axes
        # ax.set_xlabel('First Moment (Mean)')
        # ax.set_ylabel('Second Moment (Variance)')
        # ax.set_zlabel('Third Moment (Skewness)')

        # # Color bar to indicate cluster labels
        # colorbar = plt.colorbar(scatter, ax=ax)
        # colorbar.set_label('Cluster Labels')

        # # Title and grid
        # ax.set_title('3D Plot of Interarrival Time Moments with K-Means Clustering')
        # ax.grid(True)
        # plt.show()

        return cluster_dict, cluster_intervals

    time_buckets = create_time_buckets()
    fully_dy_dataset = fully_dynamic_buckets(result, time_buckets)
    # print(len(fully_dy_dataset))
    clusters_dict, clusters_intervals = plot_moments_with_clusters(fully_dy_dataset, num_k)
    # plot_combined_with_clusters(hourly_counts, clusters_intervals, num_k, interval_hours)
    # plot_combined_colored_intervals(hourly_counts, clusters_intervals, num_k, interval_hours)


    return clusters_dict, clusters_intervals, fully_dy_dataset











def simulate_arrival_times_with_clusters(cluster_intervals, start_date, cluster_dict, time_generated_arrivals):
    '''
    Generate arrival times using change point algorithm and clusters, until the total time of all arrivals reaches a specified duration.
    
    Parameters:
        cluster_intervals (dict): Mapping of cluster labels to their active time intervals.
        start_date (str or datetime): The start date and time for generating arrivals.
        cluster_dict (dict): Mapping of clusters to their corresponding interarrival times.
        time_generated_arrivals (int): Duration in seconds for which to generate arrivals from the start_date.
    '''
    # Initialize the simulated times list with the start date
    base_year = pd.to_datetime(start_date).year  # Use the year from the start_date
    current_time = pd.to_datetime(start_date)    # Use start_date directly for initialization
    simulated_dates = [current_time]
    interarrival_times = [0]

    # Calculate the end time for the entire simulation
    end_time = current_time + pd.Timedelta(seconds=time_generated_arrivals)

    # Organize clusters by their start times as specified in cluster_intervals
    time_sorted_clusters = []
    for label, intervals in cluster_intervals.items():
        for start, end in intervals:
            start = pd.to_datetime(start).replace(year=base_year)  # Normalize the year
            end = pd.to_datetime(end).replace(year=base_year)
            time_sorted_clusters.append((start, end, label))
    
    # Sort clusters based on the start time of their intervals
    time_sorted_clusters.sort()

    for start, end, numeric_label in time_sorted_clusters:
        cluster_key = next((key for key in cluster_dict if f"cluster number: {numeric_label}.0," in key), None)

        if cluster_key and not cluster_dict[cluster_key].empty:
            interarrivals = cluster_dict[cluster_key]['interarrival_time'].dropna().values

            while current_time <= end and current_time < end_time:
                sampled_interarrival = np.random.choice(interarrivals)
                next_time = current_time + pd.Timedelta(seconds=sampled_interarrival)
                
                if next_time > end_time:
                    break  # Stop generating if the next time exceeds the allowed simulation period
                
                simulated_dates.append(next_time)
                interarrival_times.append(sampled_interarrival)
                current_time = next_time

        else:
            # If no data available, skip this cluster
            print(f"Skipping cluster {numeric_label} due to no data available.")
            current_time = end + pd.Timedelta(seconds=1)
            if current_time > end_time:
                break  # Exit if current time exceeds the period

    # Create the DataFrame for the simulated arrival times
    simulated_df = pd.DataFrame({
        'arrival_date': simulated_dates,
        'interarrival_time': interarrival_times
    })

    return simulated_df










def generate_arrivals_from_dynamic_buckets(dynamic_buckets, start_date, time_generated_arrivals):
    '''
    Generate arrival times based on dynamic bucket intervals until a specified time duration is reached.
    
    Parameters:
        dynamic_buckets (dict): Dictionary with time intervals as keys and DataFrames with interarrival times as values.
        start_date (str): The starting point for generating arrivals.
        time_generated_arrivals (int): Total time in seconds for which to generate arrivals from the start_date.
    '''

    base_year = pd.to_datetime(start_date).year  # Use the year from the start_date
    current_time = pd.to_datetime(start_date)    # Use start_date directly for initialization
    
    end_time = current_time + pd.Timedelta(seconds=time_generated_arrivals)  # End time for generation
    
    # Prepare lists to hold the result
    arrival_dates = [current_time]
    interarrival_times = [0]  # Start with an interarrival time of 0
    
    # Process each bucket in the dynamic_buckets
    for bucket_interval, bucket_data in dynamic_buckets.items():
        # Parse the interval string and replace the year for proper comparison
        start_str, end_str = bucket_interval.split(" to ")
        bucket_start_time = pd.to_datetime(start_str).replace(year=base_year)
        bucket_end_time = pd.to_datetime(end_str).replace(year=base_year)

        if not bucket_data.empty:
            bucket_interarrival_times = bucket_data['interarrival_time'].dropna().to_numpy()

            # Generate arrivals while the current time is within the span of the simulation
            while current_time <= bucket_end_time and current_time < end_time:
                # Sample an interarrival time and calculate the next arrival
                sampled_interarrival = np.random.choice(bucket_interarrival_times)
                next_time = current_time + pd.Timedelta(seconds=sampled_interarrival)
                
                if next_time > end_time:
                    break  # Stop generating if the next time exceeds the allowed simulation period
                
                # Append the next arrival and the sampled interarrival time
                arrival_dates.append(next_time)
                interarrival_times.append(sampled_interarrival)
                
                # Update the current time
                current_time = next_time

                # Break if we reach the end of the day
                if next_time > bucket_end_time:
                    break
                
    # Create a DataFrame from the collected data
    simulated_arrivals = pd.DataFrame({
        'arrival_date': arrival_dates,
        'interarrival_time': interarrival_times
    })

    return simulated_arrivals








def calculate_max_date(cluster_dict, start_datetime, time_generated_arrivals):
    # Adjust the end time based on the provided duration in seconds
    return start_datetime + pd.Timedelta(seconds=time_generated_arrivals)

def simulate_arrival_times_randomly(cluster_dict, start_date, time_generated_arrivals):
    '''
    Generate arrival times completely randomly, random sampling all the interarrival times with replacement,
    up to a specified duration in seconds from the start_date.

    Parameters:
        cluster_dict (dict): Dictionary of DataFrames, each containing 'interarrival_time' for a specific cluster.
        start_date (str): Start date from which to begin simulation.
        time_generated_arrivals (int): Duration in seconds for which to simulate arrivals from the start_date.
    '''
    start_datetime = pd.to_datetime(start_date).normalize()

    # Adjust maximum date to the end time based on the total duration
    max_date = calculate_max_date(cluster_dict, start_datetime, time_generated_arrivals)

    simulated_dates = [start_datetime]  # Start from the start_datetime
    interarrival_times = [0]  # Initial interarrival time is zero

    all_interarrivals = np.concatenate([df['interarrival_time'].dropna().values for df in cluster_dict.values() if not df.empty])

    current_time = start_datetime

    # Generate arrivals until the current_time reaches the max_date
    while current_time < max_date:
        sampled_interarrival = np.random.choice(all_interarrivals)
        current_time += pd.Timedelta(seconds=sampled_interarrival)

        # Ensure the generated time does not exceed max_date
        if current_time <= max_date:
            simulated_dates.append(current_time)
            interarrival_times.append(sampled_interarrival)
        else:
            break

    # Create a DataFrame from the collected data
    arrival_df = pd.DataFrame({
        'arrival_date': simulated_dates,
        'interarrival_time': interarrival_times
    })

    return arrival_df







def simulate_arrival_times_with_random_change_points(data, interval_hours, n_bkps):
    '''
    Generate arrivals with random breakpoints, without using change point algo, no clustering
    '''
    # Ensure 'arrival_date' is in datetime format and resample data by the specified interval
    data['arrival_date'] = pd.to_datetime(data['arrival_date'])
    hourly_counts = data.resample(f'{interval_hours}h', on='arrival_date').size()

    # Generate random breakpoints, ensuring no duplicates and sorted
    breakpoints = sorted(np.random.choice(range(1, len(hourly_counts)), n_bkps, replace=False))
    breakpoints = [0] + breakpoints + [len(hourly_counts)]  # Include start and end
    '''
    # Plotting the hourly data
    plt.figure(figsize=(14, 7))
    plt.plot(hourly_counts.index, hourly_counts, linestyle='-', color='blue', label='Hourly Arrivals')

    # Annotate the random breakpoints directly with plt.axvline
    for i, bk in enumerate(breakpoints[1:-1]):  # Skip first and last as they are not actual breakpoints
        plt.axvline(x=hourly_counts.index[bk], color='red', linestyle='--', label='Random Breakpoint' if i == 0 else "")

    # Setting x-axis ticks and formatting
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=15))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.title(f'Arrivals Every {interval_hours} Hours with Random Divisions')
    plt.xlabel('Date and Time')
    plt.ylabel('Number of Arrivals')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    '''

    def create_time_buckets():
        # Ensure 'arrival_date' is in datetime format
        data['arrival_date'] = pd.to_datetime(data['arrival_date'])

        # Initialize the dictionary to hold the data buckets
        time_buckets = {}

        # Get the start and end of the dataset
        start_date = data['arrival_date'].min().floor('h')  # Round down to the nearest hour
        end_date = data['arrival_date'].max().ceil('h')  # Round up to the nearest hour

        # Generate buckets from start_date to end_date with step of 'interval_hours'
        current_time = start_date
        while current_time < end_date:
            # Define the end of the current bucket
            next_time = current_time + pd.Timedelta(hours=interval_hours)

            # Create the bucket key in the format "YYYY-MM-DD HH:MM-HH:MM"
            bucket_key = f"{current_time.strftime('%Y-%m-%d %H:%M')}-{next_time.strftime('%H:%M')}"

            # Filter data within the current time bucket
            mask = (data['arrival_date'] >= current_time) & (data['arrival_date'] < next_time)
            filtered_data = data.loc[mask]

            # Store the bucket if there is data
            if not filtered_data.empty:
                time_buckets[bucket_key] = filtered_data[['arrival_date', 'interarrival_time']]

            # Move to the next time bucket
            current_time = next_time
        
        # print(time_buckets)

        return time_buckets

    def fully_dynamic_buckets(L_breakpoints, arrival_data):
        dynamic_buckets = {}
        start_index = 0

        times = list(arrival_data.keys())
        for i, end_index in enumerate(L_breakpoints + [len(times)]):
            if start_index < len(times) and end_index <= len(times):
                start_time = times[start_index]
                end_time = times[end_index - 1] if end_index < len(times) else times[-1]

                # Change here: Include the entire datetime string for start and end times
                start_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2})-\d{2}:\d{2}', start_time)
                if start_match:
                    start_time_formatted = start_match.group(1)  # Extracts 'YYYY-MM-DD HH:MM' from the start time string

                # Extract and conditionally adjust the end time
                end_match = re.match(r'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})-(\d{2}:\d{2})', end_time)
                if end_match:
                    date_part = end_match.group(1)  # Extract the date part '2012-11-22'
                    end_time = end_match.group(3)  # Extract the end time '00:00' or other times

                    # Convert date_part to a datetime object
                    date_obj = datetime.strptime(date_part, '%Y-%m-%d')

                    if end_time == '00:00':
                        # Since time is '00:00', increment the date to the next day
                        next_day = date_obj + timedelta(days=1)
                        result_str = next_day.strftime('%Y-%m-%d') + ' ' + end_time  # 'YYYY-MM-DD 00:00'
                    else:
                        # Regular time, return same day with end time
                        result_str = date_part + ' ' + end_time

                # Create the bucket key
                bucket_key = f"{start_time_formatted} to {result_str}"
                bucket_frames = [arrival_data.get(time) for time in times[start_index:end_index] if time in arrival_data]
                if bucket_frames:
                    dynamic_buckets[bucket_key] = pd.concat(bucket_frames)
                # else:
                #     dynamic_buckets[bucket_key] = pd.DataFrame()
                # std = round(pd.concat(bucket_frames)['interarrival_time'].std(), 3)
                # mean = round(pd.concat(bucket_frames)['interarrival_time'].mean(), 3)
                # stats.append({
                #     'bucket_key': bucket_key,
                #     'std': std,
                #     'mean': mean
                # })
                start_index = end_index

        for key, df in dynamic_buckets.items():
            if not df.empty and 'interarrival_time' in df.columns:
                df['Mean_Interarrival'] = df['interarrival_time'].mean()
                df['Variance_Interarrival'] = df['interarrival_time'].var()
                df['Skewness_Interarrival'] = df['interarrival_time'].skew()

        # print(dynamic_buckets)
        return dynamic_buckets
    
    time_buckets = create_time_buckets()
    dynamic_buckets = fully_dynamic_buckets(breakpoints, time_buckets)

    return dynamic_buckets
