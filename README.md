
# Efficient Data Stream Anomaly Detection

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Algorithm Selection](#algorithm-selection)
    - [Isolation Forest (Selected)](#isolation-forest-selected)
    - [Other Approaches Considered](#other-approaches-considered)
4. [Comparative Analysis of Selected Algorithms](#comparative-analysis-of-selected-algorithms)
5. [File Structure](#file-structure)
6. [Detailed File Descriptions](#detailed-file-descriptions)
7. [Installation](#installation)
8. [How to Run](#how-to-run)
9. [Testing](#testing)
10. [Performance Metrics](#performance-metrics)
11. [Error Handling](#error-handling)
12. [Future Improvements](#future-improvements)
13. [License](#license)

## Project Overview

This project was developed as part of the application process for the Graduate Software Engineer role at Cobblestone Energy. The goal of the project is to detect anomalies in a continuous data stream, simulating real-time sequences of floating-point numbers. These data streams could represent various metrics, such as financial transactions or system metrics. The primary objective is to accurately identify unusual patterns, such as exceptionally high values or deviations from the norm.

Key features of the project include:
- The use of **Isolation Forest** for anomaly detection, which adapts to concept drift and seasonal variations.
- The generation of a synthetic data stream that simulates realistic behaviors like trends, seasonality, noise, drift, and anomalies.
- Real-time anomaly detection and data visualization to display both the data stream and the detected anomalies.
- Robust error handling, data validation, and unit tests to ensure stability and accuracy.

---

## Features

- **Anomaly Detection with Isolation Forest**: The project leverages the Isolation Forest algorithm for detecting anomalies in a real-time data stream. It is well-suited for unsupervised anomaly detection, where no labeled data is available.
- **Data Stream Simulation**: A function generates a continuous data stream with trends, seasonality, noise, drift, and anomalies.
- **Real-Time Visualization**: The project provides real-time graphical representations of the data stream, highlighting the anomalies detected by the Isolation Forest model and potential drift points.
- **Drift Detection**: Drift detection adjusts the model when concept drift (shifts in the data distribution) occurs.
- **Scalability**: The data is preprocessed using scaling techniques to improve model performance.
- **Robust Testing**: Unit tests ensure the accuracy and stability of the project components.

---

## Algorithm Selection

### Isolation Forest (Selected)
The **Isolation Forest** algorithm was selected for this project based on the following key advantages:
1. **Unsupervised Learning**: No need for labeled data, making it ideal for anomaly detection in real-time streams.
2. **Handling Concept Drift**: Isolation Forest adapts well to changes in the data over time.
3. **Efficiency**: Its tree-based approach isolates outliers efficiently, suitable for real-time detection tasks.
4. **Robustness**: It is resilient to high-dimensional data and noise, making it versatile for future extensions like multivariate data streams.

### Other Approaches Considered
Several other algorithms were considered but eventually discarded due to their limitations:

#### Sliding Window with Standard Deviation (Discarded)
- **Pros**: Simple to implement and effective for stable data.
- **Cons**: Sensitive to concept drift and does not handle seasonal variations well.
- **Reason for Discarding**: Not adaptable to changes in the data, making it unsuitable for dynamic data streams.

#### Fixed Threshold Based on Percentiles (Discarded)
- **Pros**: Easy to implement and understand.
- **Cons**: Does not adapt to real-time data changes, leading to high false positive or negative rates.
- **Reason for Discarding**: Lack of adaptability made it ineffective for dynamic and complex data streams.

#### Autoencoders (Discarded)
- **Pros**: Can detect complex patterns and subtle anomalies.
- **Cons**: Requires longer training time and more computational resources.
- **Reason for Discarding**: Autoencoders were not ideal due to the time constraints and computational complexity required for this project.

#### Dynamic Threshold Based on Median (Selected for Comparison)
- **Pros**: Adapts to gradual changes in the data and is computationally efficient.
- **Cons**: Not as advanced as Isolation Forest for detecting complex anomalies.
- **Reason for Selection**: The dynamic threshold was tested as a quick and effective method to adapt to real-time data changes.

---

## Comparative Analysis of Selected Algorithms

Both **Isolation Forest** and the **Dynamic Threshold** methods were tested and compared using the following criteria:

1. **Precision & Recall**: Measured how many real anomalies were detected by each algorithm and how many false positives were generated.
2. **Execution Speed**: Evaluated the runtime of each approach in processing the data stream.
3. **Adaptability to Concept Drift**: Measured how well the algorithms adapted to changes in the data stream.
4. **Interpretability**: Assessed how easy it was to understand and interpret the results of each algorithm.

### Results

| Algorithm          | Precision | Recall | Execution Time |
|--------------------|-----------|--------|----------------|
| Isolation Forest   | 0.9800    | 0.9800 | 0.1170 seconds |
| Dynamic Threshold  | 0.0799    | 0.9540 | 0.0437 seconds |

- **Conclusion**: Although the Dynamic Threshold approach was faster, its low precision made it less suitable for this project. Isolation Forest offered a better balance between precision and execution time, making it the ideal choice for anomaly detection in real-time data streams.

---

## File Structure

```
efficient_anomaly_detection/
├── README.md               # This documentation file
├── main.py                 # Main script that ties all the components together
├── anomaly_detector.py     # Implements Isolation Forest for anomaly detection
├── data_generator.py       # Generates the synthetic data stream
├── data_scaler.py          # Scales the data for better model performance
├── drift_detector.py       # Detects drift points in the data stream
├── plotter.py              # Visualizes the data stream with anomalies and drift points
├── utils.py                # Utility functions for logging and metrics
├── test_project.py         # Unit tests for various components
├── requirements.txt        # Required Python libraries
```

---

## Detailed File Descriptions

1. **main.py**: 
   - The entry point for the project. It generates the data stream, detects anomalies, scales the data, and visualizes the results in real-time.

2. **anomaly_detector.py**:
   - Implements the Isolation Forest algorithm for detecting anomalies in the data stream, with methods for fitting the model and predicting anomalies.

3. **data_generator.py**:
   - Generates a continuous data stream with trends, seasonality, noise, and drift. Anomalies are introduced at random intervals to simulate real-world scenarios.

4. **data_scaler.py**:
   - Scales the data using StandardScaler to normalize it before anomaly detection.

5. **drift_detector.py**:
   - Detects drift points based on abrupt changes in the data's rolling mean.

6. **plotter.py**:
   - Provides real-time visualization of the data stream, detected anomalies, and drift points.

7. **utils.py**:
   - Contains utility functions such as `log_error` for error logging and `calculate_metrics` for calculating true positives, false positives, and false negatives.

8. **test_project.py**:
   - Includes unit tests for validating the correctness of data generation, scaling, anomaly detection, and drift detection.

---

## Installation

To set up and run this project, follow these steps:

1. **Clone the repository**:
   ```
   git clone https://github.com/PabloSimonEstrada/efficient_anomaly_detection.git
   cd efficient_anomaly_detection
   ```

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

---

## How to Run

To generate the data stream, detect anomalies, and visualize the results in real-time, run the following command:
```
python main.py
```
You will see a real-time graph of the data stream, anomalies detected, and drift points updated in batches to simulate live streaming data.

---

## Testing

Unit tests are included to verify the correctness of the anomaly detection, data generation, scaling, and drift detection functions. To run the tests:
```
python test_project.py
```

The tests check:
- Correct generation of the data stream.
- Proper scaling of the data.
- Accurate detection of anomalies and drift points.
- Handling of edge cases (e.g., empty data streams).

---

## Performance Metrics

This project calculates the following metrics for evaluating the anomaly detection model:

- **True Positives (TP)**: Correctly detected anomalies.
- **False Positives (FP)**: Normal points incorrectly classified as anomalies.
- **False Negatives (FN)**: Anomalies that were missed by the model.

These metrics are printed at the end of the execution for review.

---

## Error Handling

The project includes robust error handling. Any exceptions during execution are logged using the `log_error` function from the `utils.py` module, ensuring the system remains stable and recoverable.

---

## Future Improvements

Although the project meets the requirements, there are several potential improvements:

- **Multivariate Time Series**: Extend the anomaly detection to handle multivariate data streams.
- **Advanced Drift Detection**: Implement more sophisticated drift detection methods.
- **Performance Optimization**: Further optimize the real-time detection system for larger datasets.

---

## License

This project was developed as part of an interview process for Cobblestone Energy and is subject to their intellectual property guidelines.

---

