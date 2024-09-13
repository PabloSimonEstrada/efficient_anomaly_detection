from sklearn.preprocessing import StandardScaler


def scale_data(data_stream):
    """
    Scales the data stream using standard normalization.

    This function scales the input data stream to have a mean of 0 and a standard deviation of 1, which is useful
    for improving the performance of anomaly detection models such as Isolation Forest.

    :param data_stream: Array-like object representing the data stream values.
    :return: A scaled version of the data stream with shape (-1, 1).
    :raises ValueError: If the input data stream is empty.
    """
    if len(data_stream) == 0:
        raise ValueError("The data stream is empty.")

    # Initialize the StandardScaler to normalize the data stream
    scaler = StandardScaler()

    # Reshape the data and apply the scaling transformation
    return scaler.fit_transform(data_stream.reshape(-1, 1))
