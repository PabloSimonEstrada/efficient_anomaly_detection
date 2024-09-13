import numpy as np

def detect_simple_drift(data_stream, window_size=50, drift_threshold=0.2):
    """
    Detects drift points in a data stream based on abrupt changes in the rolling mean.

    This function uses a simple drift detection method based on calculating the rolling mean of the data stream
    and identifying points where the difference between consecutive rolling mean values exceeds a given threshold.
    These points are considered as drift points, indicating potential shifts in the underlying data distribution.

    :param data_stream: Array-like object representing the data stream values.
    :param window_size: The size of the moving window used to calculate the rolling mean (default is 50).
    :param drift_threshold: The threshold that determines how large the difference between consecutive rolling mean values
                            must be to consider it as drift (default is 0.2).
    :return: A NumPy array of indices where drift was detected.
    """
    drift_points = []  # List to store the indices where drift is detected

    # Calculate the rolling mean of the data stream with the specified window size
    rolling_mean = np.convolve(data_stream, np.ones(window_size) / window_size, mode='valid')

    # Compare consecutive rolling mean values to detect drift
    for i in range(1, len(rolling_mean)):
        # If the absolute difference between consecutive rolling means exceeds the threshold, record the drift point
        if np.abs(rolling_mean[i] - rolling_mean[i - 1]) > drift_threshold:
            drift_points.append(i + window_size)

    return np.array(drift_points)
