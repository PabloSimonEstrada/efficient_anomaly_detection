from anomaly_detector import IsolationForestAnomalyDetector
from data_generator import generate_advanced_data_stream
from data_scaler import scale_data
from drift_detector import detect_simple_drift
from plotter import plot_real_time
from utils import log_error, calculate_metrics
import logging

if __name__ == "__main__":
    try:
        # Generate a data stream with predefined parameters such as noise level, trend factor, and seasonality
        data_stream, true_anomalies = generate_advanced_data_stream(
            num_points=1000,
            noise_level=0.05,
            trend_factor=0.001,
            seasonality_period=150,
            anomaly_freq=0.04,
            anomaly_magnitude=4
        )

        # Detect drift points in the data stream using a simple drift detection method
        drift_points = detect_simple_drift(data_stream)

        # Scale the data stream using StandardScaler for better anomaly detection performance
        data_stream_scaled = scale_data(data_stream)

        # Initialize the Isolation Forest anomaly detector
        detector = IsolationForestAnomalyDetector(contamination=0.05, n_estimators=200)

        # Fit the Isolation Forest model to the scaled data stream
        detector.fit(data_stream_scaled)

        # Predict anomalies in the data stream
        if_anomalies = detector.predict(data_stream_scaled)

        # If drift points are detected, log the event and update the model accordingly
        if len(drift_points) > 0:
            logging.info("Drift detected, updating the model...")
            detector.update_model(data_stream_scaled)

        # Visualize the real-time data stream along with detected anomalies and drift points
        plot_real_time(
            data_stream,
            if_anomalies,
            drift_points,
            true_anomalies,
            update_interval=0.5,
            batch_size=10
        )

    except Exception as e:
        # Log any errors that occur during the execution
        log_error(e)

# Calculate performance metrics based on the detected and true anomalies
tp, fp, fn = calculate_metrics(if_anomalies, true_anomalies)

# Print the calculated metrics for the model's performance
print(f'True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}')
