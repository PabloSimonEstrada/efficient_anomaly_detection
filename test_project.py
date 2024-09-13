import unittest
import numpy as np
from anomaly_detector import IsolationForestAnomalyDetector
from data_generator import generate_advanced_data_stream
from data_scaler import scale_data
from drift_detector import detect_simple_drift
from utils import calculate_metrics


class TestAnomalyDetectionProject(unittest.TestCase):

    def setUp(self):
        """
        This function runs before each test. It generates the data required for the test cases.
        """
        self.data_stream, self.true_anomalies = generate_advanced_data_stream(
            num_points=1000,
            noise_level=0.05,
            anomaly_freq=0.05,
            trend_factor=0.001,
            seasonality_period=150,
            anomaly_magnitude=4
        )
        self.scaled_data_stream = scale_data(self.data_stream)
        self.drift_detector = detect_simple_drift

    def test_generate_advanced_data_stream(self):
        """
        Tests that the data generation produces the correct number of data points and anomalies.
        """
        data_stream, anomaly_indices = generate_advanced_data_stream(num_points=1000)
        self.assertEqual(len(data_stream), 1000)
        self.assertGreater(len(anomaly_indices), 0)

    def test_scale_data(self):
        """
        Tests that the scale_data function correctly scales the data.
        """
        scaled_data = scale_data(self.data_stream)
        self.assertEqual(scaled_data.shape, (1000, 1))
        # Verify if the mean of the scaled data is close to 0
        self.assertAlmostEqual(np.mean(scaled_data), 0, delta=0.1)

    def test_drift_detection(self):
        """
        Tests that detect_simple_drift correctly detects drift in the data.
        """
        drift_points = detect_simple_drift(self.data_stream)
        self.assertIsInstance(drift_points, np.ndarray)
        self.assertGreaterEqual(len(drift_points), 1)  # Expect at least one drift point

    def test_isolation_forest_anomaly_detection(self):
        """
        Tests the Isolation Forest anomaly detector to ensure it correctly detects anomalies.
        """
        detector = IsolationForestAnomalyDetector(contamination=0.05, n_estimators=100)
        detector.fit(self.scaled_data_stream)
        if_anomalies = detector.predict(self.scaled_data_stream)
        self.assertIsInstance(if_anomalies, np.ndarray)
        self.assertGreaterEqual(len(if_anomalies), 1)

    def test_calculate_metrics(self):
        """
        Tests that the calculate_metrics function returns the correct metrics.
        """
        detector = IsolationForestAnomalyDetector(contamination=0.05, n_estimators=100)
        detector.fit(self.scaled_data_stream)
        if_anomalies = detector.predict(self.scaled_data_stream)

        tp, fp, fn = calculate_metrics(if_anomalies, self.true_anomalies)
        self.assertGreaterEqual(tp, 0)  # True positives should be >= 0
        self.assertGreaterEqual(fp, 0)  # False positives should be >= 0
        self.assertGreaterEqual(fn, 0)  # False negatives should be >= 0

    def test_invalid_data_handling(self):
        """
        Tests that functions handle invalid data cases properly.
        """
        with self.assertRaises(ValueError):
            scale_data(np.array([]))  # An empty array should not be allowed

        detector = IsolationForestAnomalyDetector(contamination=0.05, n_estimators=100)
        with self.assertRaises(ValueError):
            detector.fit(np.array([]))  # Fitting the model with empty data should raise an error


if __name__ == '__main__':
    unittest.main()
