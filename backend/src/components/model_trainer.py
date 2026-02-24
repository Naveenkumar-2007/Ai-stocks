import numpy as np
import joblib
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import os
from dotenv import load_dotenv

load_dotenv()

class ModelTrainer:
    def __init__(self):
        self.model = None
        
    def build_model(self, input_shape):
        self.model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(100, return_sequences=True),
            Dropout(0.3),
            LSTM(50, return_sequences=False),
            Dropout(0.3),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    def train_model(self, X_train, y_train, X_test, y_test,
                    epochs=100, batch_size=32, callbacks=None):
        """
        Train LSTM model and return (model, history).
        """
        # Defensive shape check
        if len(X_train.shape) < 3:
            raise ValueError(f"Invalid X_train shape: {X_train.shape}. Expected 3D array (samples, seq_len, features).")

        self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Default callbacks if none provided
        if callbacks is None:
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
            ]
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=callbacks
        )
        
        # Save model
        os.makedirs('artifacts', exist_ok=True)
        self.model.save('artifacts/stock_lstm_model.h5')
        
        return self.model, history