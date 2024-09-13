import logging

def log_error(e):
    """
    Logs any error that occurs during the execution of the program.

    :param e: Exception object that contains the error details.
    """
    logging.error(f"Error detected: {e}")

def calculate_metrics(if_anomalies, true_anomalies, tolerance=5):
    """
    Calculates the true positives (TP), false positives (FP), and false negatives (FN)
    between the anomalies detected by the Isolation Forest model and the true anomalies.

    :param if_anomalies: Indices of anomalies detected by the Isolation Forest (IF) model.
    :param true_anomalies: Indices of the actual anomalies in the data.
    :param tolerance: Tolerance range (number of indices) to consider a detection as correct.
    :return: Tuple containing the counts of true positives (TP), false positives (FP), and false negatives (FN).
    """
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives

    # True Positives: Detected anomalies that match true anomalies within the tolerance range
    for anomaly in if_anomalies:
        if any(abs(anomaly - ta) <= tolerance for ta in true_anomalies):
            tp += 1
        else:
            fp += 1

    # False Negatives: True anomalies that were not detected by the model
    for true_anomaly in true_anomalies:
        if all(abs(true_anomaly - da) > tolerance for da in if_anomalies):
            fn += 1

    return tp, fp, fn
