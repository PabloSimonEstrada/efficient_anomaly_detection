import numpy as np

def generate_advanced_data_stream(num_points=1000, noise_level=0.05, anomaly_freq=0.05, trend_factor=0.001,
                                  seasonality_period=200, anomaly_magnitude=4, drift_frequency=700):
    """
    Generates a synthetic data stream with trend, seasonality, noise, drift, and anomalies.

    This function creates a data stream with several components to simulate real-world time series behavior:
    - A linear trend
    - Seasonality (sinusoidal pattern)
    - Random noise
    - Drift, which occurs periodically and shifts the baseline of the data
    - Anomalies, which are artificially introduced to simulate unusual behavior

    :param num_points: The total number of points to generate in the data stream (default is 1000).
    :param noise_level: The standard deviation of the random noise added to the data (default is 0.05).
    :param anomaly_freq: The frequency of anomalies, as a proportion of the total number of points (default is 0.05).
    :param trend_factor: The slope of the linear trend component in the data (default is 0.001).
    :param seasonality_period: The period of the sinusoidal seasonality component (default is 200).
    :param anomaly_magnitude: The magnitude of the anomalies introduced (default is 4).
    :param drift_frequency: The frequency (number of points) at which drift is introduced (default is 700).
    :return: A tuple containing:
             - data_stream: The generated data stream (NumPy array).
             - anomaly_indices: The indices of the introduced anomalies.
    """
    # Create a linear trend component
    x = np.arange(num_points)
    trend = trend_factor * np.arange(num_points)

    # Add random noise
    noise = noise_level * np.random.randn(num_points)

    # Create a seasonality component using sine and cosine functions
    seasonality = np.sin(2 * np.pi * x / seasonality_period) + np.cos(4 * np.pi * x / seasonality_period)

    # Introduce drift by periodically shifting the baseline of the data
    drift = np.zeros(num_points)
    for i in range(0, num_points, drift_frequency):
        drift[i:] += np.random.uniform(-1, 1)  # Add a random shift to the remaining data after each interval

    # Combine the trend, seasonality, noise, and drift components to form the final data stream
    data_stream = trend + seasonality + noise + drift

    # Introduce anomalies at random positions
    anomaly_indices = np.random.choice(num_points, int(num_points * anomaly_freq), replace=False)
    data_stream[anomaly_indices] += np.random.uniform(anomaly_magnitude, anomaly_magnitude * 2, size=anomaly_indices.size)

    return data_stream, anomaly_indices
