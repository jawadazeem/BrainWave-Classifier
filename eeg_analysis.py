import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load EEG-like time series data
# Expected CSV format: Time, Channel1, Channel2, ..., Condition
df = pd.read_csv("eeg_timeseries.csv")

# Separate signal data and labels
signal_data = df.drop(columns=["Time", "Condition"])
labels = df["Condition"]

# Compute FFT and extract frequency features
sampling_rate = 250  # Hz (adjust as needed)
n = len(df)
freqs = np.fft.rfftfreq(n, d=1/sampling_rate)

# Perform FFT for each channel
fft_data = np.abs(np.fft.rfft(signal_data, axis=0)) ** 2  # power spectrum

# Define EEG bands (Hz)
bands = {"Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 12), "Beta": (13, 30)}

# Compute average band power per channel
band_powers = {}
for band, (low, high) in bands.items():
    idx = np.where((freqs >= low) & (freqs <= high))[0]
    band_powers[band] = fft_data[idx, :].mean(axis=0)

# Build a features DataFrame
features = pd.DataFrame(band_powers)
features["Condition"] = labels

# Train a machine learning model
X = features.drop(columns=["Condition"])
y = features["Condition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualization
# Average band power by condition
avg_bands = features.groupby("Condition")[["Alpha", "Beta", "Theta"]].mean()

plt.figure(figsize=(8, 5))
for band in ["Alpha", "Beta", "Theta"]:
    plt.plot(avg_bands.index, avg_bands[band], marker='o', label=band)

plt.title("Average EEG Band Power by Condition")
plt.xlabel("Condition")
plt.ylabel("Power (µV²)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
