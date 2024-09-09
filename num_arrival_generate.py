import pandas as pd
import numpy as np

def simulate_arrival_times_with_clusters(cluster_intervals, start_date, cluster_dict, num_generated_arrivals):
    '''
    generate arrival times using change point algo, and clusters 
    '''
    # Initialize the simulated times list and the first entry
    base_year = pd.to_datetime(start_date).year  # Use the year from the start_date
    current_time = pd.to_datetime(start_date)    # Use start_date directly for initialization
    # print(current_time)

    simulated_dates = [current_time]
    interarrival_times = [0]
    generated_count = 1
    
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
        # print(end)
        # Find the corresponding key in cluster_dict
        cluster_key = next((key for key in cluster_dict if f"cluster number: {numeric_label}.0," in key), None)

        if cluster_key and not cluster_dict[cluster_key].empty:
            interarrivals = cluster_dict[cluster_key]['interarrival_time'].dropna().values
            # print(f"Processing cluster {numeric_label} from {start} to {end} with {len(interarrivals)} options")

            # print(interarrivals)

            while current_time <= end:
                if generated_count >= num_generated_arrivals:
                    # If the generation limit is reached, stop generating more
                    break
                sampled_interarrival = np.random.choice(interarrivals)
                next_time = current_time + pd.Timedelta(seconds=sampled_interarrival)
            
                simulated_dates.append(next_time)
                interarrival_times.append(sampled_interarrival)
                current_time = next_time
                # print(f"Sampled {sampled_interarrival}s, next arrival at {next_time}")
                generated_count += 1

                if next_time > end:
                    # Move current_time to just past the end to prevent infinite loop on cluster edge cases
                    # current_time = end + pd.Timedelta(seconds=1)
                    # print(f"Moving to next cluster after {current_time}")
                    break

        else:
            # If no data available, move to the next cluster
            print(f"Skipping cluster {numeric_label} due to no data available.")
            current_time = end + pd.Timedelta(seconds=1)

    # Create the DataFrame for the simulated arrival times
    simulated_df = pd.DataFrame({
        'arrival_date': simulated_dates,
        'interarrival_time': interarrival_times
    })

    return simulated_df











def generate_arrivals_from_dynamic_buckets(dynamic_buckets, start_date, num_generated_arrivals):

    '''
    generate arrival times straight from change point, no clusters  
    '''

    base_year = pd.to_datetime(start_date).year  # Use the year from the start_date
    # print(base_year)
    current_time = pd.to_datetime(start_date)    # Use start_date directly for initialization
    # print(current_time)
    
    # Prepare lists to hold the result
    arrival_dates = [current_time]
    interarrival_times = [0]  # Start with an interarrival time of 0
    generated_count = 1

    
    # Process each bucket in the dynamic_buckets
    for bucket_interval, bucket_data in dynamic_buckets.items():
        # Extract interarrival times from the current bucket's DataFrame
        start_str, end_str = bucket_interval.split(" to ")
        bucket_start_time = pd.to_datetime(start_str).replace(year=base_year)
        bucket_end_time = pd.to_datetime(end_str).replace(year=base_year)
        # print(bucket_end_time)

        if not bucket_data.empty:
            bucket_interarrival_times = bucket_data['interarrival_time'].dropna().to_numpy()
            # print(f"Processing from {bucket_start_time} to {bucket_end_time} with {len(bucket_interarrival_times)} options")
            # print(bucket_interarrival_times)

            # Generate arrivals while the current time is within the span of the simulation
            while current_time <= bucket_end_time:  
                if generated_count >= num_generated_arrivals:
                    # If the generation limit is reached, stop generating more
                    break
                # Sample an interarrival time and calculate the next arrival
                sampled_interarrival = np.random.choice(bucket_interarrival_times)
                next_time = current_time + pd.Timedelta(seconds=sampled_interarrival)
                
                
                # Append the next arrival and the sampled interarrival time
                arrival_dates.append(next_time)
                interarrival_times.append(sampled_interarrival)
                
                # Update the current time
                current_time = next_time
                # print(f"Sampled {sampled_interarrival}s, next arrival at {next_time}")

                generated_count += 1

                
                # Break if we reach the end of the day
                if next_time > bucket_end_time:
                    # print(f"Moving to next cluster after {current_time}")
                    break
                

    # Create a DataFrame from the collected data
    simulated_arrivals = pd.DataFrame({
        'arrival_date': arrival_dates,
        'interarrival_time': interarrival_times
    })

    return simulated_arrivals








def calculate_max_date(cluster_dict, start_datetime):
    min_date = min(df['arrival_date'].min() for df in cluster_dict.values() if not df.empty)
    max_date = max(df['arrival_date'].max() for df in cluster_dict.values() if not df.empty)
    # Compute the number of days between the max and min date
    num_days = (max_date - min_date).days + 1
    return start_datetime + pd.Timedelta(days=num_days)

def simulate_arrival_times_randomly(cluster_dict, start_date, num_generated_arrivals):
    '''
    generate arrival times completely randomly, random sampling all the interarrival times with replacement
    '''
    start_datetime = pd.to_datetime(start_date).normalize()

    # Get the maximum date based on the cluster data
    max_date = calculate_max_date(cluster_dict, start_datetime)
    # print(max_date)

    simulated_dates = [start_datetime]  # Start from the start_datetime
    interarrival_times = [0]  # Initial interarrival time is zero
    generated_count = 1

    all_interarrivals = np.concatenate([df['interarrival_time'].dropna().values for df in cluster_dict.values() if not df.empty])
    # print(len(all_interarrivals))

    current_time = start_datetime

    while current_time < max_date:
        if generated_count >= num_generated_arrivals:
            # If the generation limit is reached, stop generating more
            break
        sampled_interarrival = np.random.choice(all_interarrivals)
        current_time += pd.Timedelta(seconds=sampled_interarrival)

        if current_time <= max_date:
            simulated_dates.append(current_time)
            interarrival_times.append(sampled_interarrival)
            generated_count += 1

    # Ensure last time does not exceed max_date
    if simulated_dates[-1] > max_date:
        simulated_dates.pop()
        interarrival_times.pop()

    arrival_df = pd.DataFrame({
        'arrival_date': simulated_dates,
        'interarrival_time': interarrival_times
    })


    return arrival_df

