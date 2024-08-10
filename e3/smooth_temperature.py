import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter

file_path = sys.argv[1]

cpu_data = pd.read_csv(file_path)

cpu_data['timestamp_sec'] = (pd.to_datetime(cpu_data['timestamp']) - pd.to_datetime(cpu_data['timestamp']).min()).dt.total_seconds()

# LOESS smoothing
loess_smoothed = lowess(cpu_data['temperature'], cpu_data['timestamp_sec'], frac=0.05)[:, 1]

# Kalman smoothing
kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1', 'fan_rpm']]

initial_state = kalman_data.iloc[0]
observation_covariance = np.diag([119.48632616, 1, 1, 1]) # Utilizing variance of temperature
transition_covariance = np.diag([0.01, 0.01, 0.01, 0.01]) ** 2
transition = [[0.97, 0.5, 0.2, -0.001], [0.1, 0.4, 2.2, 0], [0, 0, 0.95, 0], [0, 0, 0, 1]]

kf = KalmanFilter(
    initial_state_mean=initial_state,
    observation_covariance=observation_covariance,
    transition_covariance=transition_covariance,
    transition_matrices=transition,
    em_vars=['transition_covariance', 'observation_covariance'])

kalman_smoothed, _ = kf.smooth(kalman_data.values)

# Plot and save the data
plt.figure(figsize=(12, 4))

plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5, label='Original Data')
plt.plot(cpu_data['timestamp'], loess_smoothed, 'r-', label='LOESS Smoothed')
plt.plot(cpu_data['timestamp'], kalman_smoothed[:, 0], 'g-', label='Kalman Smoothed')

plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.title('CPU Temperature Smoothing')
plt.legend()
plt.grid(True)

plt.savefig('cpu.svg')


# Used ChatGPT to resolve the timestamp error and coverting into seconds.
