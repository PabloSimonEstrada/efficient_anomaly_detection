import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import time

def plot_real_time(data_stream, if_anomalies, drift_points, true_anomalies=None, update_interval=0.1, batch_size=50):
    """
    Plots a real-time graph of a data stream, highlighting detected anomalies and drift points.

    This function allows you to visualize a data stream in real-time, marking the anomalies detected by
    the Isolation Forest model (if_anomalies), and any drift points identified. If true anomalies are
    provided, they will be displayed for comparison.

    :param data_stream: Array-like object containing the data stream values.
    :param if_anomalies: Array of indices representing anomalies detected by the Isolation Forest model.
    :param drift_points: Array of indices representing detected drift points.
    :param true_anomalies: (Optional) Array of indices representing the true anomalies in the data stream.
    :param update_interval: Time in seconds between each update to simulate real-time behavior.
    :param batch_size: Number of data points to be processed and displayed per batch.
    """
    plt.ion()  # Activate interactive mode for real-time animation
    fig, ax = plt.subplots(figsize=(10, 6))  # Define plot size for better visualization

    # Initialize empty lists to store data for real-time plotting
    x_data, y_data = [], []
    anomaly_x, anomaly_y = [], []
    drift_x, drift_y = [], []
    true_anomaly_x, true_anomaly_y = [], []

    # Main line plot for the data stream
    line, = ax.plot(x_data, y_data, label='Data Stream', lw=2, color='navy')

    # Scatter plot for anomalies and drift points
    anomaly_scatter = ax.scatter(anomaly_x, anomaly_y, color='red', label='IF Anomalies', s=50, alpha=0.9, edgecolor='k', zorder=3, marker='o')
    drift_scatter = ax.scatter(drift_x, drift_y, color='orange', label='Drift Points', s=50, alpha=0.5, edgecolor='k', zorder=1)

    # Scatter plot for true anomalies (if provided)
    if true_anomalies is not None:
        true_anomaly_scatter = ax.scatter(true_anomaly_x, true_anomaly_y, color='blue', label='True Anomalies', s=70, alpha=0.7, edgecolor='k', zorder=2, marker='^')

    # Setting up the title and labels for the plot
    ax.set_title('Real-Time Data Stream with Anomaly and Drift Detection', fontsize=16, fontweight='bold', color='darkblue')
    ax.set_xlabel('Index', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)

    # Customize legend for better visualization
    legend_elements = [Patch(color='red', label='IF Anomalies'),
                       Patch(color='orange', label='Drift Points'),
                       Patch(color='blue', label='True Anomalies')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=True, shadow=True)

    # Add grid lines for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Real-time statistics box for displaying mean, standard deviation, min, and max of the current batch
    stats_box = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='lightgrey', alpha=0.7))

    # Loop for processing and visualizing data in real-time
    for i in range(0, len(data_stream), batch_size):
        new_data_batch = data_stream[i:i + batch_size]
        new_indices = list(range(i, i + batch_size))

        # Update data for the main line plot
        x_data.extend(new_indices)
        y_data.extend(new_data_batch)

        line.set_xdata(x_data)
        line.set_ydata(y_data)

        # Update anomalies detected by Isolation Forest
        new_anomalies = if_anomalies[(if_anomalies >= i) & (if_anomalies < i + batch_size)]
        if new_anomalies.size > 0:
            anomaly_x.extend(new_anomalies)
            anomaly_y.extend(data_stream[new_anomalies])
            anomaly_scatter.set_offsets(np.c_[anomaly_x, anomaly_y])

        # Update detected drift points
        new_drift = drift_points[(drift_points >= i) & (drift_points < i + batch_size)]
        if new_drift.size > 0:
            drift_x.extend(new_drift)
            drift_y.extend(data_stream[new_drift])
            drift_scatter.set_offsets(np.c_[drift_x, drift_y])

            # Add a semi-transparent patch to highlight drift areas
            for drift_point in new_drift:
                ax.axvspan(drift_point - batch_size, drift_point + batch_size, color='orange', alpha=0.2, zorder=0)

        # Update true anomalies (if provided)
        if true_anomalies is not None:
            new_true_anomalies = true_anomalies[(true_anomalies >= i) & (true_anomalies < i + batch_size)]
            if new_true_anomalies.size > 0:
                true_anomaly_x.extend(new_true_anomalies)
                true_anomaly_y.extend(data_stream[new_true_anomalies])
                true_anomaly_scatter.set_offsets(np.c_[true_anomaly_x, true_anomaly_y])

        # Update real-time statistics box
        mean_value = np.mean(new_data_batch)
        std_value = np.std(new_data_batch)
        min_value = np.min(new_data_batch)
        max_value = np.max(new_data_batch)
        stats_box.set_text(f'Mean: {mean_value:.2f}\nStd: {std_value:.2f}\nMin: {min_value:.2f}\nMax: {max_value:.2f}')

        # Dynamically adjust axes for new data
        ax.relim()
        ax.autoscale_view()

        # Redraw the figure and update the UI in real-time
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Pause to simulate real-time updates
        time.sleep(update_interval)

    # Disable interactive mode after the loop is complete
    plt.ioff()
    plt.show()
