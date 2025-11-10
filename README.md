# EEG Analysis Project

This project analyzes EEG-like time series data using Python.  
It extracts frequency features using FFT (Fast Fourier Transform), trains a Random Forest model to classify conditions, and visualizes the average band power.

## Overview
1. Load EEG data from a CSV file.
2. Compute frequency features (Delta, Theta, Alpha, Beta).
3. Build a feature table with labeled data.
4. Train a Random Forest classifier.
5. Display model accuracy and a simple band power plot.


## Requirements
Install packages:
pip install -r requirements.txt

**requirements.txt**
pandas
numpy
matplotlib
scikit-learn


## Run the Script

You’ll get:
- Console output with accuracy and classification report
- A plot comparing EEG band power by condition

## Data Notice
The included CSV uses synthetic (fake) EEG-like values for demonstration.  
Do not upload real EEG data from participants or experiments publicly.

## Learning Value
- Data preprocessing and feature extraction  
- Frequency analysis with FFT  
- Machine learning model training and evaluation  
- Visualization of results

## Author
Jawad Azeem
