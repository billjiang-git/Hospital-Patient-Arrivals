## main.py

import pandas as pd
import numpy as np
from utils import read_data
from error_funcs import (compute_mse_between_dataframes, compute_dtw_between_dataframes, compute_emd_between_dataframes
)
from time_arrival_generate import (
    change_point_on_dataset, simulate_arrival_times_with_clusters, generate_arrivals_from_dynamic_buckets,  
    simulate_arrival_times_randomly, simulate_arrival_times_with_random_change_points
)
from plotting_arrivals import (
    plotting_arrivals, plotting_arrivals_all_plots, plotting_arrivals_combine, plot_combined_colored_intervals, 
    plot_combined_with_clusters
)
from data_processing import adjust_datasets_to_shortest_length
import matplotlib.pyplot as plt


def main():
    ## Read and process data
    file_path = "/Users/billjiang/Desktop/event_log_Singapore.xlsx"
    df = read_data(file_path)
    # print(df['arrival_date'])






    # ####### Split the data based on number of arrivals ########
    # total_rows = len(df)  # Total number of rows in the DataFrame

    # # Calculate the indices for the splits
    # train_end = int(total_rows * 0.6)  # End of train dataset
    # validation_end = train_end + int(total_rows * 0.2)  # End of validation dataset

    # # Split the data into train, validation, and test sets
    # train_df = df.iloc[:train_end]
    # validation_df = df.iloc[train_end:validation_end]
    # test_df = df.iloc[validation_end:]

    # print(train_df)
    # print(validation_df)
    # print(test_df)

    ####### Split the data based on time ########
    # Calculate the cutoff times for each dataset split
    total_time = df['arrival_date'].max() - df['arrival_date'].min()
    train_end_time = df['arrival_date'].min() + total_time * 0.6
    validation_end_time = df['arrival_date'].min() + total_time * 0.8

    # Create masks for splitting the data
    train_mask = df['arrival_date'] <= train_end_time
    validation_mask = (df['arrival_date'] > train_end_time) & (df['arrival_date'] <= validation_end_time)
    test_mask = df['arrival_date'] > validation_end_time

    # Split the data into train, validation, and test sets
    train_df = df[train_mask]
    validation_df = df[validation_mask]
    test_df = df[test_mask]

    # Calculate the total time for each set
    train_time = (train_df['arrival_date'].max() - train_df['arrival_date'].min()).total_seconds()
    validation_time = (validation_df['arrival_date'].max() - validation_df['arrival_date'].min()).total_seconds()
    test_time = (test_df['arrival_date'].max() - test_df['arrival_date'].min()).total_seconds()


    # Print out the shape of the splits to confirm the process
    # print(train_df)
    # print(validation_df)
    # # print(test_df)
    # print(f"Train set time: {train_time}")
    # print(f"Validation set time: {validation_time}")
    # print(f"Test set time: {test_time}")


    data = df
    hour_interval = 1
    num_k = 10
    bkps = 50
    start_date = '2021-11-02'  # Start date for the simulation
    # num_generated_arrivals = len(test_df)
    time_generated_arrivals = validation_time


    ##### GROUND TRUTH #####

    ground_truth_entire_dataset = data[['arrival_date', 'interarrival_time']].copy()
    ground_truth_entire_dataset.fillna(0, inplace=True)
    # print(ground_truth_entire_dataset)
    # plot_for_ground_truth_entire_dataset = plotting_arrivals(ground_truth_entire_dataset, hour_interval, bkps, 'Ground Truth')

    ground_truth_train_dataset = train_df[['arrival_date', 'interarrival_time']].copy()
    ground_truth_train_dataset.fillna(0, inplace=True)
    # plot_for_ground_truth_train_dataset = plotting_arrivals(ground_truth_train_dataset, hour_interval, bkps, 'Ground Truth')


    ground_truth_validation_dataset = validation_df[['arrival_date', 'interarrival_time']].copy()
    ground_truth_validation_dataset.fillna(0, inplace=True)
    # plot_for_ground_truth_validation_dataset = plotting_arrivals(ground_truth_validation_dataset, hour_interval, bkps, 'Ground Truth')


    ground_truth_test_dataset = test_df[['arrival_date', 'interarrival_time']].copy()
    ground_truth_test_dataset.fillna(0, inplace=True)
    # plot_for_ground_truth_test_dataset = plotting_arrivals(ground_truth_test_dataset, hour_interval, bkps, 'Ground Truth')




    ###### Generate the arrivals with change points and clusters ######

    cluster_dict, cluster_intervals, fully_dy_dataset_with_change_points = change_point_on_dataset(train_df.copy(), hour_interval, num_k, bkps)
    # print(cluster_dict)
    # print(cluster_intervals)


    fully_dy_dataset_without_change_points = simulate_arrival_times_with_random_change_points(data, hour_interval, bkps)
    generated_arrivals_with_random_change_points_no_k_means = generate_arrivals_from_dynamic_buckets(fully_dy_dataset_without_change_points, start_date, time_generated_arrivals)
    # plot_for_generated_arrivals_with_random_change_points_no_k_means = plotting_arrivals(generated_arrivals_with_random_change_points_no_k_means, hour_interval, bkps, 'Random Change Points No K Means')
    # print(generated_arrivals_with_random_change_points_no_k_means)


    generated_arrivals_with_change_points_no_k_means = generate_arrivals_from_dynamic_buckets(fully_dy_dataset_with_change_points, start_date, time_generated_arrivals)
    # plot_for_generated_arrivals_with_change_points_no_k_means = plotting_arrivals(generated_arrivals_with_change_points_no_k_means, hour_interval, bkps, 'Change Points No K Means')
    # print(generated_arrivals_with_change_points_no_k_means)


    simulated_arrivals_random = simulate_arrival_times_randomly(cluster_dict, start_date, time_generated_arrivals)
    # plot_for_simulated_arrivals_random = plotting_arrivals(simulated_arrivals_random, hour_interval, bkps, 'Random')
    # print(simulated_arrivals_random)


    simulated_arrivals_with_clusters = simulate_arrival_times_with_clusters(cluster_intervals, start_date, cluster_dict, time_generated_arrivals)
    # plot_for_simulated_arrivals_with_clusters = plotting_arrivals(simulated_arrivals_with_clusters, hour_interval, bkps, 'Clustered')
    # print(simulated_arrivals_with_clusters)




    def adjust_ground_truth_to_match_dates_and_times(ground_truth_validation_dataset, start_date):
        start_date_to_datetime = pd.to_datetime(start_date).normalize()
        current_time = start_date_to_datetime
        generated_times =  [start_date_to_datetime]
        interarrival_times = ground_truth_validation_dataset['interarrival_time'].values
        for i in range(1, len(interarrival_times)):
            current_time = current_time + pd.Timedelta(seconds=interarrival_times[i])
            generated_times.append(current_time)
        ground_truth_validation_dataset['arrival_date'] = generated_times
        return ground_truth_validation_dataset

    ground_truth_validation_dataset = adjust_ground_truth_to_match_dates_and_times(ground_truth_validation_dataset, start_date)
    ground_truth_test_dataset = adjust_ground_truth_to_match_dates_and_times(ground_truth_test_dataset, start_date)
    # print(ground_truth_validation_dataset)










    
    # # ######## ALL PLOTS SHOWN SEPARATELY ON ONE ########

    # fig, axs = plt.subplots(2, 1, figsize=(12, 8))  # Adjust size and layout to 4x1

    # # Plot individually on separate subplots
    # plotting_arrivals_all_plots(ground_truth_validation_dataset, hour_interval, 'Ground Truth', axs[0])
    # plotting_arrivals_all_plots(generated_arrivals_with_change_points_no_k_means, hour_interval, 'Change Points', axs[1])
    # # plotting_arrivals_all_plots(simulated_arrivals_with_clusters, hour_interval, 'Clustered', axs[2])
    # # plotting_arrivals_all_plots(generated_arrivals_with_random_change_points_no_k_means, hour_interval, 'Random Change Points', axs[3])
    # # plotting_arrivals_all_plots(simulated_arrivals_random, hour_interval, 'Random', axs[4])


    # # Adjust layout and display the combined figure
    # plt.tight_layout()
    # plt.show()

    
    
    # ############## ALL PLOTS COMBINED ON ONE ###########
    # # Create a single figure and axes
    # fig, ax = plt.subplots(figsize=(14, 7))

    # # print(ground_truth_test_dataset)
    # # Plot each dataset on the same axes
    # # plotting_arrivals_combine(generated_arrivals_with_random_change_points_no_k_means, hour_interval, 'Random Change Points', ax)
    # plotting_arrivals_combine(generated_arrivals_with_change_points_no_k_means, hour_interval, 'Change Points', ax)
    # # plotting_arrivals_combine(simulated_arrivals_random, hour_interval, 'Random', ax)
    # plotting_arrivals_combine(simulated_arrivals_with_clusters, hour_interval, 'Clustered', ax)
    # plotting_arrivals_combine(ground_truth_validation_dataset, hour_interval, 'Ground Truth', ax)

    # # Set a common title and show the plot
    # ax.set_title('Combined Arrival Data Visualization')
    # plt.tight_layout()
    # plt.show()

    
    
    # # # ######## MSE ###########

    # list_of_dfs = [
    #     ground_truth_validation_dataset,
    #     simulated_arrivals_with_clusters, 
    #     simulated_arrivals_random, 
    #     generated_arrivals_with_random_change_points_no_k_means, 
    #     generated_arrivals_with_change_points_no_k_means, 
    # ]
    # # print(ground_truth_test_dataset, simulated_arrivals_with_clusters)

    # adjusted_dfs = adjust_datasets_to_shortest_length(list_of_dfs)

    # # print(adjusted_dfs)

    # mse_result_simulated_with_clusters = compute_mse_between_dataframes(adjusted_dfs[0], adjusted_dfs[1])
    # # print(adjusted_dfs[0], adjusted_dfs[1])
    # mse_result_simulated_randomly = compute_mse_between_dataframes(adjusted_dfs[0], adjusted_dfs[2])
    # mse_result_simulated_with_random_change_points_no_k_means = compute_mse_between_dataframes(adjusted_dfs[0], adjusted_dfs[3])
    # mse_result_simulated_with_change_points_no_k_means = compute_mse_between_dataframes(adjusted_dfs[0], adjusted_dfs[4])

    # mse_results = {
    #     "Simulated with Clusters": mse_result_simulated_with_clusters,
    #     "Simulated Randomly": mse_result_simulated_randomly,
    #     "Simulated with Random Change Points (No K-Means)": mse_result_simulated_with_random_change_points_no_k_means,
    #     "Simulated with Change Points (No K-Means)": mse_result_simulated_with_change_points_no_k_means
    # }

    # min_mse_type, min_mse_value = min(mse_results.items(), key=lambda x: x[1])
    # print("MSE Results for Different Simulation Types:")
    # print(f"1. MSE - Simulated with Clusters: {mse_result_simulated_with_clusters:.2e}")
    # print(f"2. MSE - Simulated Randomly: {mse_result_simulated_randomly:.2e}")
    # print(f"3. MSE - Simulated with Random Change Points (No K-Means): {mse_result_simulated_with_random_change_points_no_k_means:.2e}")
    # print(f"4. MSE - Simulated with Change Points (No K-Means): {mse_result_simulated_with_change_points_no_k_means:.2e}")


    # print(f"\nMinimum MSE Result:")
    # print(f"{min_mse_type}: {min_mse_value:.2e}")





    ######## EMD ###########
    # # Calculate EMD for different pairs
    # emd_result_simulated_with_clusters = compute_emd_between_dataframes(ground_truth_validation_dataset, simulated_arrivals_with_clusters)
    # emd_result_simulated_randomly = compute_emd_between_dataframes(ground_truth_validation_dataset, simulated_arrivals_random)
    # emd_result_simulated_with_random_change_points_no_k_means = compute_emd_between_dataframes(ground_truth_validation_dataset, generated_arrivals_with_random_change_points_no_k_means)
    # emd_result_simulated_with_change_points_no_k_means = compute_emd_between_dataframes(ground_truth_validation_dataset, generated_arrivals_with_change_points_no_k_means)


    # emd_results = {
    #     "Simulated with Clusters": emd_result_simulated_with_clusters,
    #     "Simulated Randomly": emd_result_simulated_randomly,
    #     "Simulated with Random Change Points (No K-Means)": emd_result_simulated_with_random_change_points_no_k_means,
    #     "Simulated with Change Points (No K-Means)": emd_result_simulated_with_change_points_no_k_means
    # }

    # # Find the minimum EMD value and the corresponding simulation type
    # min_emd_type, min_emd_value = min(emd_results.items(), key=lambda x: x[1])

    # # Print EMD results in a similar formatted style as MSE
    # print("EMD Results for Different Simulation Types:")
    # print(f"1. EMD - Simulated with Clusters: {emd_result_simulated_with_clusters:.2e}")
    # print(f"2. EMD - Simulated Randomly: {emd_result_simulated_randomly:.2e}")
    # print(f"3. EMD - Simulated with Random Change Points (No K-Means): {emd_result_simulated_with_random_change_points_no_k_means:.2e}")
    # print(f"4. EMD - Simulated with Change Points (No K-Means): {emd_result_simulated_with_change_points_no_k_means:.2e}")


    # print(f"\nMinimum EMD Result:")
    # print(f"{min_emd_type}: {min_emd_value:.2e}")








    ######### DTW ###########

    # # Calculate DTW for different pairs
    # dtw_result_simulated_with_clusters = compute_dtw_between_dataframes(ground_truth_validation_dataset, simulated_arrivals_with_clusters, 'arrival_date')
    # dtw_result_simulated_randomly = compute_dtw_between_dataframes(ground_truth_validation_dataset, simulated_arrivals_random, 'arrival_date')
    # dtw_result_simulated_with_random_change_points_no_k_means = compute_dtw_between_dataframes(ground_truth_validation_dataset, generated_arrivals_with_random_change_points_no_k_means, 'arrival_date')
    # dtw_result_simulated_with_change_points_no_k_means = compute_dtw_between_dataframes(ground_truth_validation_dataset, generated_arrivals_with_change_points_no_k_means, 'arrival_date')

    # # Store the DTW results in a dictionary
    # dtw_results = {
    #     "Simulated with Clusters": dtw_result_simulated_with_clusters,
    #     "Simulated Randomly": dtw_result_simulated_randomly,
    #     "Simulated with Random Change Points (No K-Means)": dtw_result_simulated_with_random_change_points_no_k_means,
    #     "Simulated with Change Points (No K-Means)": dtw_result_simulated_with_change_points_no_k_means
    # }

    # # Find the minimum DTW value and the corresponding simulation type
    # min_dtw_type, min_dtw_value = min(dtw_results.items(), key=lambda x: x[1])

    # # Print DTW results in a formatted style
    # print("DTW Results for Different Simulation Types:")
    # print(f"1. DTW - Simulated with Clusters: {dtw_result_simulated_with_clusters:.2e}")
    # print(f"2. DTW - Simulated Randomly: {dtw_result_simulated_randomly:.2e}")
    # print(f"3. DTW - Simulated with Random Change Points (No K-Means): {dtw_result_simulated_with_random_change_points_no_k_means:.2e}")
    # print(f"4. DTW - Simulated with Change Points (No K-Means): {dtw_result_simulated_with_change_points_no_k_means:.2e}")

    # print(f"\nMinimum DTW Result:")
    # print(f"{min_dtw_type}: {min_dtw_value:.2e}")
















    # # Define a range of values for breakpoints and number of clusters
    # bkps_range = [i for i in range (20, 31, 5)]  # Example ranges for breakpoints
    # num_k_range = [i for i in range (1, 51, 10)]  # Example ranges for number of clusters


# # The lowest mse is 1.93e+07 found with 49 clusters and 59 breakpoints




#     ######## Best num_k for every bkps MSE ########
#     # Dictionary to store the results
#     results = {}

#     # Loop over all combinations of bkps and num_k
#     for bkps in bkps_range:
#         results[bkps] = {}
#         for num_k in num_k_range:
#             # Generate the dataset with change points and clusters
#             cluster_dict, cluster_intervals, fully_dy_dataset_with_change_points = change_point_on_dataset(
#                 train_df.copy(), hour_interval, num_k, bkps)
            
#             if not cluster_intervals:  # Check if cluster_intervals is empty

#                 # print(f"No cluster intervals generated for bkps: {bkps}, num_k: {num_k}")
                
#                 continue  # Skip this iteration

#             simulated_arrivals_with_clusters = simulate_arrival_times_with_clusters(
#                 cluster_intervals, start_date, cluster_dict, time_generated_arrivals)
            
#             # Compute mse with the ground truth dataset
#             mse_result = compute_mse_between_dataframes(ground_truth_validation_dataset, simulated_arrivals_with_clusters, 'arrival_date')
            
#             # Store the result in the dictionary
#             results[bkps][num_k] = mse_result

#             # Print current combination and its mse for tracking progress
#             print(f"Breakpoints: {bkps}, Clusters: {num_k}, mse: {mse_result:.2e}")

#     # Find the best combination of bkps and num_k for each bkps setting
#     for bkps, mses in results.items():
#         best_num_k = min(mses, key=mses.get)  # Get the num_k with the minimum mse
#         print(f"Best num_k for {bkps} breakpoints is {best_num_k} with mse {mses[best_num_k]:.2e}")

#     min_mse = float('inf')
#     best_num_k = None
#     best_bkps = None

#     for bkps, mses in results.items():
#         for num_k, mse in mses.items():
#             if mse < min_mse:
#                 min_mse = mse
#                 best_num_k = num_k
#                 best_bkps = bkps

#     # Output the results
#     print(f"The lowest mse is {min_mse:.2e} found with {best_num_k} clusters and {best_bkps} breakpoints.")




    # # Define a range of values for breakpoints and number of clusters
    # bkps_range = [i for i in range (40, 66, 5)]  # Example ranges for breakpoints
    # num_k_range = [i for i in range (1, 51, 10)]  # Example ranges for number of clusters


    # ######## Best num_k for every bkps EMD ########
    # # Dictionary to store the results
    # results = {}

    # # Loop over all combinations of bkps and num_k
    # for bkps in bkps_range:
    #     results[bkps] = {}
    #     for num_k in num_k_range:
    #         # Generate the dataset with change points and clusters
    #         cluster_dict, cluster_intervals, fully_dy_dataset_with_change_points = change_point_on_dataset(
    #             train_df.copy(), hour_interval, num_k, bkps)
            
    #         if not cluster_intervals:  # Check if cluster_intervals is empty

    #             # print(f"No cluster intervals generated for bkps: {bkps}, num_k: {num_k}")
                
    #             continue  # Skip this iteration

    #         simulated_arrivals_with_clusters = simulate_arrival_times_with_clusters(
    #             cluster_intervals, start_date, cluster_dict, time_generated_arrivals)
            
    #         # Compute EMD with the ground truth dataset
    #         emd_result = compute_emd_between_dataframes(ground_truth_validation_dataset, simulated_arrivals_with_clusters, 'arrival_date')
            
    #         # Store the result in the dictionary
    #         results[bkps][num_k] = emd_result

    #         # Print current combination and its EMD for tracking progress
    #         print(f"Breakpoints: {bkps}, Clusters: {num_k}, EMD: {emd_result:.2f}")

    # # Find the best combination of bkps and num_k for each bkps setting
    # for bkps, emds in results.items():
    #     best_num_k = min(emds, key=emds.get)  # Get the num_k with the minimum EMD
    #     print(f"Best num_k for {bkps} breakpoints is {best_num_k} with EMD {emds[best_num_k]:.2f}")

    # min_emd = float('inf')
    # best_num_k = None
    # best_bkps = None

    # for bkps, emds in results.items():
    #     for num_k, emd in emds.items():
    #         if emd < min_emd:
    #             min_emd = emd
    #             best_num_k = num_k
    #             best_bkps = bkps

    # # Output the results
    # print(f"The lowest emd is {min_emd:.2f} found with {best_num_k} clusters and {best_bkps} breakpoints.")



    # Define a range of values for breakpoints and number of clusters
    bkps_range = range(40, 66, 5)  # Example ranges for breakpoints
    num_k_range = range(1, 51, 10)  # Example ranges for number of clusters

    # Function to execute the main process
    def main_simulation(n):
        # Dictionary to store cumulative results for averaging
        cumulative_results = {bkps: {num_k: [] for num_k in num_k_range} for bkps in bkps_range}

        # Run the process n times
        for _ in range(n):
            # Dictionary to store the results of the current run
            results = {bkps: {} for bkps in bkps_range}
            
            for bkps in bkps_range:
                for num_k in num_k_range:
                    # Placeholder functions for dataset operations (to be implemented)
                    cluster_dict, cluster_intervals, fully_dy_dataset_with_change_points = change_point_on_dataset(
                        train_df.copy(), hour_interval, num_k, bkps)
                    
                    if not cluster_intervals:
                        continue  # Skip if no clusters generated

                    simulated_arrivals_with_clusters = simulate_arrival_times_with_clusters(
                        cluster_intervals, start_date, cluster_dict, time_generated_arrivals)
                    
                    emd_result = compute_emd_between_dataframes(ground_truth_validation_dataset, simulated_arrivals_with_clusters, 'arrival_date')
                    results[bkps].setdefault(num_k, []).append(emd_result)

            # Accumulate results from the current run
            for bkps in results:
                for num_k in results[bkps]:
                    cumulative_results[bkps][num_k].extend(results[bkps][num_k])

        # Calculate the average EMD for each combination
        min_emd = float('inf')
        best_bkps = None
        best_num_k = None
        
        for bkps in cumulative_results:
            for num_k in cumulative_results[bkps]:
                avg_emd = np.mean(cumulative_results[bkps][num_k])
                if avg_emd < min_emd:
                    min_emd = avg_emd
                    best_bkps = bkps
                    best_num_k = num_k

        # Output the best results
        print(f"The lowest EMD is {min_emd:.2f} found with {best_num_k} clusters and {best_bkps} breakpoints.")
        return best_bkps, best_num_k, min_emd


    # Example usage
    n_runs = 10  # Number of times to run the simulation
    average_results = main_simulation(n_runs)
    # print(average_results)
    best_bkps, best_num_k, min_emd = average_results









    # # Define a range of values for breakpoints and number of clusters
    # bkps_range = [i for i in range (15, 21, 1)]  # Example ranges for breakpoints
    # num_k_range = [i for i in range (1, 51, 10)]  # Example ranges for number of clusters


    # ######## Best bkps for every num_k EMD ########

    # # Dictionary to store the results
    # results = {num_k: {} for num_k in num_k_range}

    # # Loop over all combinations of num_k and bkps
    # for num_k in num_k_range:
    #     for bkps in bkps_range:
    #         # Generate the dataset with change points and clusters
    #         cluster_dict, cluster_intervals, fully_dy_dataset_with_change_points = change_point_on_dataset(
    #             train_df, hour_interval, num_k, bkps)
            
    #         if not cluster_intervals:  # Check if cluster_intervals is empty

    #             # print(f"No cluster intervals generated for bkps: {bkps}, num_k: {num_k}")
                
    #             continue  # Skip this iteration

    #         simulated_arrivals_with_clusters = simulate_arrival_times_with_clusters(
    #             cluster_intervals, start_date, cluster_dict, time_generated_arrivals)
            
    #         # Compute EMD with the ground truth dataset
    #         emd_result = compute_emd_between_dataframes(ground_truth_test_dataset, simulated_arrivals_with_clusters, 'arrival_date')
            
    #         # Store the result in the dictionary
    #         results[num_k][bkps] = emd_result

    #         # Print current combination and its EMD for tracking progress
    #         print(f"Clusters: {num_k}, Breakpoints: {bkps}, EMD: {emd_result:.2f}")

    # # Find the best bkps for each num_k setting
    # for num_k, emds in results.items():
    #     best_bkps = min(emds, key=emds.get)  # Get the bkps with the minimum EMD
    #     print(f"Best breakpoints for {num_k} clusters is {best_bkps} with EMD {emds[best_bkps]:.2f}")

    # min_emd = float('inf')
    # best_num_k = None
    # best_bkps = None

    # # Loop through results to find the lowest EMD
    # for num_k, emds in results.items():
    #     for bkps, emd in emds.items():
    #         if emd < min_emd:
    #             min_emd = emd
    #             best_num_k = num_k
    #             best_bkps = bkps

    # # Output the results
    # print(f"The lowest EMD is {min_emd:.2f} found with {best_num_k} clusters and {best_bkps} breakpoints.")








    # ######## Best bkps for every num_k MSE ########

    # # Dictionary to store the results
    # results = {num_k: {} for num_k in num_k_range}

    # # Loop over all combinations of num_k and bkps
    # for num_k in num_k_range:
    #     for bkps in bkps_range:
    #         # Generate the dataset with change points and clusters
    #         cluster_dict, cluster_intervals, fully_dy_dataset_with_change_points = change_point_on_dataset(
    #             data, hour_interval, num_k, bkps)
            
    #         if not cluster_intervals:  # Check if cluster_intervals is empty

    #             # print(f"No cluster intervals generated for bkps: {bkps}, num_k: {num_k}")
                
    #             continue  # Skip this iteration

    #         simulated_arrivals_with_clusters = simulate_arrival_times_with_clusters(
    #             cluster_intervals, start_date, cluster_dict, time_generated_arrivals)
            
    #         # Compute mse with the ground truth dataset
    #         mse_result = compute_mse_between_dataframes(ground_truth_test_dataset, simulated_arrivals_with_clusters, 'arrival_date')
            
    #         # Store the result in the dictionary
    #         results[num_k][bkps] = mse_result

    #         # Print current combination and its mse for tracking progress
    #         print(f"Clusters: {num_k}, Breakpoints: {bkps}, mse: {mse_result:.2e}")

    # # Find the best bkps for each num_k setting
    # for num_k, mses in results.items():
    #     best_bkps = min(mses, key=mses.get)  # Get the bkps with the minimum mse
    #     print(f"Best breakpoints for {num_k} clusters is {best_bkps} with mse {mses[best_bkps]:.2e}")

    # min_mse = float('inf')
    # best_num_k = None
    # best_bkps = None

    # # Loop through results to find the lowest mse
    # for num_k, mses in results.items():
    #     for bkps, mse in mses.items():
    #         if mse < min_mse:
    #             min_mse = mse
    #             best_num_k = num_k
    #             best_bkps = bkps

    # # Output the results
    # print(f"The lowest mse is {min_mse:.2e} found with {best_num_k} clusters and {best_bkps} breakpoints.")

    best_num_k = 21
    best_bkps = 40

    cluster_dict, cluster_intervals, fully_dy_dataset_with_change_points = change_point_on_dataset(data, hour_interval, best_num_k, best_bkps)
    # print(cluster_dict)
    # print(cluster_intervals)


    fully_dy_dataset_without_change_points = simulate_arrival_times_with_random_change_points(data, hour_interval, best_bkps)
    generated_arrivals_with_random_change_points_no_k_means = generate_arrivals_from_dynamic_buckets(fully_dy_dataset_without_change_points, start_date, time_generated_arrivals)
    # plot_for_generated_arrivals_with_random_change_points_no_k_means = plotting_arrivals(generated_arrivals_with_random_change_points_no_k_means, hour_interval, bkps, 'Random Change Points No K Means')
    # print(fully_dy_dataset_without_change_points)


    generated_arrivals_with_change_points_no_k_means = generate_arrivals_from_dynamic_buckets(fully_dy_dataset_with_change_points, start_date, time_generated_arrivals)
    # plot_for_generated_arrivals_with_change_points_no_k_means = plotting_arrivals(generated_arrivals_with_change_points_no_k_means, hour_interval, bkps, 'Change Points No K Means')
    # print(generated_arrivals_with_change_points_no_k_means)


    simulated_arrivals_random = simulate_arrival_times_randomly(cluster_dict, start_date, time_generated_arrivals)
    # plot_for_simulated_arrivals_random = plotting_arrivals(simulated_arrivals_random, hour_interval, bkps, 'Random')
    # print(simulated_arrival_dates_random)


    simulated_arrivals_with_clusters = simulate_arrival_times_with_clusters(cluster_intervals, start_date, cluster_dict, time_generated_arrivals)
    # plot_for_simulated_arrivals_with_clusters = plotting_arrivals(simulated_arrivals_with_clusters, hour_interval, bkps, 'Clustered')
    # print(simulated_arrivals_with_clusters)





    # import numpy as np
    # from scipy.stats import mannwhitneyu

    # # Example data arrays
    # results_clustering = np.array([emd1, emd2, emd3, ...])  # Replace with actual EMD/MSE results
    # results_random = np.array([emd1_r, emd2_r, emd3_r, ...])  # Replace with actual EMD/MSE results

    # # Perform the Mann-Whitney U Test
    # stat, p_value = mannwhitneyu(results_clustering, results_random, alternative='two-sided')

    # print(f'Statistics={stat:.3f}, p={p_value:.5f}')
    # if p_value < 0.05:
    #     print('Statistically significant differences between the two methods.')
    # else:
    #     print('No statistically significant differences found between the two methods.')



    
    # # ######## ALL PLOTS SHOWN SEPARATELY ON ONE ########

    # fig, axs = plt.subplots(4, 1, figsize=(12, 8))  # Adjust size and layout to 4x1

    # # Plot individually on separate subplots
    # plotting_arrivals_all_plots(ground_truth_validation_dataset, hour_interval, 'Ground Truth Validation', axs[0])
    # plotting_arrivals_all_plots(ground_truth_test_dataset, hour_interval, 'Ground Truth Test', axs[1])
    # plotting_arrivals_all_plots(generated_arrivals_with_change_points_no_k_means, hour_interval, 'Change Points', axs[2])
    # plotting_arrivals_all_plots(simulated_arrivals_with_clusters, hour_interval, 'Clustered', axs[3])
    # # plotting_arrivals_all_plots(generated_arrivals_with_random_change_points_no_k_means, hour_interval, 'Random Change Points', axs[3])
    # # plotting_arrivals_all_plots(simulated_arrivals_random, hour_interval, 'Random', axs[4])


    # # Adjust layout and display the combined figure
    # plt.tight_layout()
    # plt.show()







    # The lowest EMD is 4961.88 found with 40 clusters and 100 breakpoints.
    # The lowest EMD is 5670.84 found with 50 clusters and 100 breakpoints.
    # The lowest EMD is 5284.62 found with 30 clusters and 50 breakpoints.
    # The lowest EMD is 5286.46 found with 60 clusters and 50 breakpoints.
    # The lowest EMD is 4935.80 found with 80 clusters and 20 breakpoints.






#     ######## BEST BKPS FOR GENERATING WITH ONLY CHANGE POINTS NO K MEANS ########
#     # Define the range for breakpoints
#     bkps_range = [10, 20, 30, 40, 50, 60]

#     # Dictionary to store the EMD results for each bkps
#     emd_results = {}

#     # Iterate over the range of bkps values
#     for bkps in bkps_range:
#         # Generate the dataset with dynamic change points
#         cluster_dict, cluster_intervals, fully_dy_dataset_with_change_points = change_point_on_dataset(
#             data, hour_interval, num_k, bkps)
        
#         # Simulate arrival times from the dynamically generated buckets
#         simulated_arrivals_with_change_points = generate_arrivals_from_dynamic_buckets(
#             fully_dy_dataset_with_change_points, start_date, num_generated_arrivals)
        
#         # Compute EMD with the ground truth dataset
#         emd_result = compute_emd_between_dataframes(ground_truth_test_dataset, simulated_arrivals_with_change_points, 'arrival_date')
        
#         # Store the EMD result
#         emd_results[bkps] = emd_result
        
#         # Print the current bkps and its EMD for tracking
#         # print(f"Breakpoints: {bkps}, EMD: {emd_result:.2f}")

#     # Find the bkps with the minimum EMD
#     best_bkps = min(emd_results, key=emd_results.get)
#     min_emd = emd_results[best_bkps]

#     # Output the best bkps and the corresponding EMD
#     print(f"The best number of breakpoints is {best_bkps} with the lowest EMD of {min_emd:.2f}.")


#     cluster_dict, cluster_intervals, fully_dy_dataset_with_change_points = change_point_on_dataset(data, hour_interval, num_k, best_bkps)
#     # print(cluster_dict)
#     # print(cluster_intervals)


#     fully_dy_dataset_without_change_points = simulate_arrival_times_with_random_change_points(data, hour_interval, best_bkps)
#     generated_arrivals_with_random_change_points_no_k_means = generate_arrivals_from_dynamic_buckets(fully_dy_dataset_without_change_points, start_date, num_generated_arrivals)
#     # plot_for_generated_arrivals_with_random_change_points_no_k_means = plotting_arrivals(generated_arrivals_with_random_change_points_no_k_means, hour_interval, bkps, 'Random Change Points No K Means')
#     # print(fully_dy_dataset_without_change_points)


#     generated_arrivals_with_change_points_no_k_means = generate_arrivals_from_dynamic_buckets(fully_dy_dataset_with_change_points, start_date, num_generated_arrivals)
#     # plot_for_generated_arrivals_with_change_points_no_k_means = plotting_arrivals(generated_arrivals_with_change_points_no_k_means, hour_interval, bkps, 'Change Points No K Means')
#     # print(generated_arrivals_with_change_points_no_k_means)


#     simulated_arrivals_random = simulate_arrival_times_randomly(cluster_dict, start_date, num_generated_arrivals)
#     # plot_for_simulated_arrivals_random = plotting_arrivals(simulated_arrivals_random, hour_interval, bkps, 'Random')
#     # print(simulated_arrival_dates_random)


#     simulated_arrivals_with_clusters = simulate_arrival_times_with_clusters(cluster_intervals, start_date, cluster_dict, num_generated_arrivals)
#     # plot_for_simulated_arrivals_with_clusters = plotting_arrivals(simulated_arrivals_with_clusters, hour_interval, bkps, 'Clustered')
#     # print(simulated_arrivals_with_clusters)


#     ######## ALL PLOTS SHOWN SEPARATELY ON ONE ########
#     # fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # Adjust size as needed

#     datasets = [simulated_arrivals_with_clusters, generated_arrivals_with_change_points_no_k_means, ground_truth_test_dataset, simulated_arrivals_random]
#     generate_types = ["Clustered", "Change Points (No K-Means)", "Ground Truth", "Random"]
#     global_y_max = 0
#     for data in datasets:
#         hourly_counts = data.resample(f'{hour_interval}h', on='arrival_date').size()
#         max_y = hourly_counts.max()
#         if max_y > global_y_max:
#             global_y_max = max_y

#     # Step 2: Plot each dataset with the same y-axis limit
#     fig, axs = plt.subplots(len(datasets), 1, figsize=(14, 7 * len(datasets)))

#     for i, (data, gen_type) in enumerate(zip(datasets, generate_types)):
#         plotting_arrivals_all_plots(data, hour_interval, gen_type, ax=axs[i], y_max=global_y_max)

#     plt.tight_layout()
#     plt.show()




#     # # Plot individually on separate subplots
#     # # plotting_arrivals_all_plots(generated_arrivals_with_random_change_points_no_k_means, hour_interval, 'Random Change Points', axs[0, 0])
#     # plotting_arrivals_all_plots(generated_arrivals_with_change_points_no_k_means, hour_interval, 'Change Points', axs[1, 1])
#     # plotting_arrivals_all_plots(ground_truth_test_dataset, hour_interval, 'Ground Truth', axs[0, 1])
#     # plotting_arrivals_all_plots(simulated_arrivals_random, hour_interval, 'Random', axs[1, 0])
#     # plotting_arrivals_all_plots(simulated_arrivals_with_clusters, hour_interval, 'Clustered', axs[0, 0])

#     # # Adjust layout and display the combined figure
#     # plt.tight_layout()
#     # plt.show()









if __name__ == "__main__":
    main()



















# ##### TO DO #####
# 3. Test EMD with Next Month Data
# 6. Coverage distance measure, run arrival generator many times, gives mean and sd of arrivals, 
# construct upper and lower bound, what percentage of the time the ground truth captures
# 7. Order cluster based on mean arrival rate
# 8. smaller dataset, or split the graph
# 9. capture two days 
# 11. repeat the random tests many times and compute the standard deviation and mean of the EMD
# 12. Train test validation, on validation, optimize bkps and clusters, after I get the best ones, test on test set
# 13. statistical significance, meaning between EMD of 10k and 5k
# 14. if results are close, choose smaller number of clusters and bkps


### Questions ###
# 1. Do I generate a certain number of arrivals or generate arrivals for a certain period? (Train test split)





# ##### DONE #####
# 1. Tidy code
# 2. Training Testing Split 60,20,20
# 5. Generate n number of arrivals, not necessarily the same amount as original data for training and testing split

