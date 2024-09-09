import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import ruptures as rpt








def plotting_arrivals(arrival_data, interval_hours, breakpoints, generate_type):
    arrival_data['arrival_date'] = arrival_data['arrival_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    arrival_data['arrival_date'] = pd.to_datetime(arrival_data['arrival_date'])
    hourly_counts = arrival_data.resample(f'{interval_hours}h', on='arrival_date').size()
    
    algo = rpt.Binseg(model="l2").fit(hourly_counts.values)
    n = len(hourly_counts)
    sigma = np.std(hourly_counts.values)
    penalty = np.log(np.log(np.log(n))) * sigma**2  # Adjust penalty as needed
    penalty = 1
    bkps = 100             # basically the number of buckets
    result = algo.predict(pen=penalty)
    result = algo.predict(n_bkps=breakpoints)
    # print(len(result))

    # Plotting the arrivals data using plt directly
    plt.figure(figsize=(14, 7))
    plt.plot(hourly_counts.index, hourly_counts, linestyle='-', color='blue', label='Hourly Arrivals')

    # Annotating the detected change points directly with plt.axvline
    for bk in result[:-1]:  # Exclude the last point as it is just the end of the data
        if bk < len(hourly_counts):  # Ensure the breakpoint index is within the range of the plot
            plt.axvline(x=hourly_counts.index[bk], color='red', linestyle='--', label='Change Point' if bk == result[0] else "")

    ##### Setting x-axis ticks and formatting
    plt.xticks(hourly_counts.index, rotation=45)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())  # Major ticks per day
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.title(f'{generate_type} Generated Arrival Data: Time vs. Arrivals per {interval_hours} Hours')
    plt.xlabel('Date')
    plt.ylabel('Number of Arrivals')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plotting_arrivals_all_plots(arrival_data, interval_hours, generate_type, ax=None):
    arrival_data['arrival_date'] = pd.to_datetime(arrival_data['arrival_date'])
    hourly_counts = arrival_data.resample(f'{interval_hours}h', on='arrival_date').size()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(hourly_counts.index, hourly_counts, linestyle='-', linewidth=1.25, label=f'{generate_type} Hourly Arrivals')
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    ax.set_title(f'{generate_type} Generated Arrival Data: Time vs. Arrivals per {interval_hours} Hours')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Arrivals')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if ax is None:
        plt.tight_layout()
        plt.show()



def plotting_arrivals_all_plots(arrival_data, interval_hours, generate_type, ax=None, y_max=None, remove_x_axis=True, remove_title=True):
    arrival_data['arrival_date'] = pd.to_datetime(arrival_data['arrival_date'])
    hourly_counts = arrival_data.resample(f'{interval_hours}h', on='arrival_date').size()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(hourly_counts.index, hourly_counts, linestyle='-', linewidth=1.25, label=f'{generate_type} Hourly Arrivals')
    
    if not remove_x_axis:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax.set_xticklabels([])  # Remove x-axis labels
        ax.set_xticks([])  # Remove x-axis ticks
    
    if not remove_title:
        ax.set_title(f'{generate_type} Generated Arrival Data: Time vs. Arrivals per {interval_hours} Hours')
    else:
        ax.set_title('')  # Remove the title
    
    ax.set_title(f'{generate_type} Generated Arrival Data: Time vs. Arrivals per {interval_hours} Hours')
    ax.set_xlabel('Date' if not remove_x_axis else '')
    ax.set_ylabel('Number of Arrivals')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Force the y-axis limit to be consistent across all plots
    if y_max is not None:
        ax.set_ylim(0, y_max)
    
    if ax is None:
        plt.tight_layout()
        plt.show()




def plotting_arrivals_combine(arrival_data, interval_hours, generate_type, ax):
    arrival_data['arrival_date'] = pd.to_datetime(arrival_data['arrival_date'])
    hourly_counts = arrival_data.resample(f'{interval_hours}h', on='arrival_date').size()
    
    # Plot without x-axis details
    ax.plot(hourly_counts, linestyle='-', linewidth=1.25, label=f'{generate_type} Hourly Arrivals')

    # Hide x-axis
    ax.xaxis.set_visible(False)

    ax.set_ylabel('Number of Arrivals')
    
    # Set y-axis grid only
    ax.yaxis.grid(True, alpha=0.3)  # Enable grid on y-axis
    ax.xaxis.grid(False)            # Disable grid on x-axis
    
    ax.legend()




def plot_combined_colored_intervals(hourly_counts, cluster_intervals, num_k, interval_hours):
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plotting hourly arrivals
    ax1.plot(hourly_counts.index, hourly_counts, 'b-', label='Hourly Arrivals')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Arrivals', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.xaxis.set_major_locator(mdates.DayLocator())  # Major ticks per day
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Creating a second Y-axis for clusters
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cluster Number', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(-1, num_k)  # Extend the limits for cluster numbers from -1 to num_k
    ax2.set_yticks(range(num_k))  # Set a tick for each cluster number

    # Plotting cluster intervals as vertical spans
    color_map = cm.get_cmap('tab10', num_k)  # Use a colormap for different clusters
    for label, intervals in cluster_intervals.items():
        for start_date, end_date in intervals:
            ax1.axvspan(start_date, end_date, color=color_map(label), alpha=0.3, label=f'Cluster {label}' if start_date == intervals[0][0] else "")

    # Consolidating legends from both axes and moving the legend outside of the plot area
    handles, labels = [], []
    for ax in [ax1, ax2]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)
    ax2.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 1), title='Legend')

    plt.title(f'Arrivals Every {interval_hours} Hours and Change Points Over Dataset')
    plt.tight_layout()  # Adjust layout to make room for the legend outside the plot
    plt.show()



def plot_combined_with_clusters(hourly_counts, cluster_intervals, num_k, interval_hours):
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plotting hourly arrivals
    ax1.plot(hourly_counts.index, hourly_counts, 'b-', label='Hourly Arrivals')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Arrivals', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.xaxis.set_major_locator(mdates.DayLocator())  # Major ticks per day
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Creating a second Y-axis for clusters
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cluster Number', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(-1, num_k)  # Extend the limits for cluster numbers from -1 to num_k
    ax2.set_yticks(range(num_k))  # Set a tick for each cluster number

    # Plotting cluster intervals
    color_map = cm.get_cmap('tab10', num_k)  # Use a colormap for different clusters
    for label, intervals in cluster_intervals.items():
        for start_date, end_date in intervals:
            ax2.plot([start_date, end_date], [label]*2, color=color_map(label), 
                     marker='|', markersize=10, label=f'Cluster {label}' if start_date == intervals[0][0] else "")

    # Consolidating legends from both axes and moving the legend outside of the plot area
    handles, labels = [], []
    for ax in [ax1, ax2]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)
    ax2.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 1), title='Legend')

    plt.title(f'Arrivals Every {interval_hours} Hours and Change Points Over Dataset')
    plt.tight_layout()  # Adjust layout to make room for the legend outside the plot
    plt.show()
