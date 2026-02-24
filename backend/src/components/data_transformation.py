import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

class DataTransformation:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def create_sequences(self, data, seq_length=60):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length, 0])
        return np.array(X), np.array(y)
    
    def transform(self, train_data, test_data, seq_length=60):
        """
        Transform train/test DataFrames into scaled LSTM-ready sequences.
        Called by training_pipeline.py.
        """
        train_values = train_data.values if hasattr(train_data, 'values') else train_data
        test_values = test_data.values if hasattr(test_data, 'values') else test_data
        
        # Fit scaler on training data only
        train_scaled = self.scaler.fit_transform(train_values)
        test_scaled = self.scaler.transform(test_values)
        
        X_train, y_train = self.create_sequences(train_scaled, seq_length)
        X_test, y_test = self.create_sequences(test_scaled, seq_length)
        
        return X_train, y_train, X_test, y_test, self.scaler