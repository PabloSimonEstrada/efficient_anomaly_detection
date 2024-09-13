from sklearn.ensemble import IsolationForest
import numpy as np


class IsolationForestAnomalyDetector:
    """
    A wrapper class for the Isolation Forest algorithm, designed to detect anomalies in a data stream.

    The Isolation Forest algorithm is an unsupervised learning method that identifies outliers by isolating them
    in the data. This class allows you to fit the model to a data stream, predict anomalies, and update the model
    when new data is available.
    """

    def __init__(self, contamination=0.05, n_estimators=100):
        """
        Initializes the IsolationForestAnomalyDetector with the specified contamination level and number of estimators.

        :param contamination: The proportion of outliers in the data. Must be a value between 0 and 0.5.
        :param n_estimators: The number of trees in the forest (default is 100).
        :raises ValueError: If the contamination value is not within the valid range (0, 0.5].
        """
        if not 0 < contamination <= 0.5:
            raise ValueError("Contamination must be between 0 and 0.5.")

        # Initialize the Isolation Forest model
        self.model = IsolationForest(contamination=contamination, n_estimators=n_estimators)

    def fit(self, data_stream):
        """
        Fits the Isolation Forest model to the provided data stream.

        :param data_stream: Array-like object representing the data stream.
        :raises ValueError: If the input data stream is empty.
        """
        if len(data_stream) == 0:
            raise ValueError("The data stream is empty.")

        # Fit the Isolation Forest model to the reshaped data stream
        self.model.fit(data_stream.reshape(-1, 1))

    def predict(self, data_stream):
        """
        Predicts anomalies in the provided data stream using the fitted Isolation Forest model.

        :param data_stream: Array-like object representing the data stream.
        :return: A NumPy array of indices where anomalies were detected.
        """
        # Get predictions from the Isolation Forest model (-1 indicates an anomaly)
        predictions = self.model.predict(data_stream.reshape(-1, 1))
        # Return the indices where anomalies were detected
        anomalies = np.where(predictions == -1)[0]
        return anomalies

    def update_model(self, data_stream):
        """
        Updates the Isolation Forest model with new data.

        This method refits the model with the new data stream, which can be used when concept drift or other
        changes are detected in the data.

        :param data_stream: Array-like object representing the new data stream.
        """
        # Refit the Isolation Forest model with the new data stream
        self.model.fit(data_stream.reshape(-1, 1))
