# predictor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import pickle
import os

def create_sequences(data, sequence_length):
    """
    Creates sequences from time-series data.
    For a sequence of length 3, it will take items 0,1,2 to predict 3,
    then 1,2,3 to predict 4, and so on.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def train_and_save_model():
    """
    Main function to load data, build, train, and save the LSTM model.
    """
    try:
        # 1. Load the realistic usage data
        df = pd.read_csv('real_usage_data.csv')
        print(f"Loaded {len(df)} data points from real_usage_data.csv")
        
        # 2. Extract CPU usage data
        cpu_data = df['cpu_usage'].values
        
        # 3. Scale the data to be between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        cpu_data_scaled = scaler.fit_transform(cpu_data.reshape(-1, 1)).flatten()
        
        # 4. Create sequences for training
        sequence_length = 12  # Use 12 time points to predict the next one
        X, y = create_sequences(cpu_data_scaled, sequence_length)
        
        print(f"Created {len(X)} sequences for training")
        
        # 5. Split data into training and testing sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # 6. Reshape data for LSTM (samples, time steps, features)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # 7. Build the LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        
        # 8. Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        # 9. Train the model
        print("Training the model...")
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # 10. Save the trained model
        model.save('cpu_predictor_model.keras')
        print("Model saved as cpu_predictor_model.keras")
        
        # 11. Save the scaler for future use
        with open('cpu_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("Scaler saved as cpu_scaler.pkl")
        
        # 12. Evaluate the model
        train_loss = model.evaluate(X_train, y_train, verbose=0)
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Training Loss: {train_loss[0]:.4f}")
        print(f"Test Loss: {test_loss[0]:.4f}")
        
        return model, scaler, history
        
    except FileNotFoundError:
        print("Error: The file 'real_usage_data.csv' was not found.")
        return None, None, None
    except Exception as e:
        print(f"Error during training: {e}")
        return None, None, None

def predict_cpu_usage(sequence_data, model_path='cpu_predictor_model.keras', scaler_path='cpu_scaler.pkl'):
    """
    Predict CPU usage for the next time step given a sequence of historical data.
    
    Args:
        sequence_data: List or array of the last 12 CPU usage values
        model_path: Path to the trained model
        scaler_path: Path to the saved scaler
    
    Returns:
        Predicted CPU usage value
    """
    try:
        # Load the trained model
        model = load_model(model_path)
        
        # Load the scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Scale the input data
        sequence_scaled = scaler.transform(np.array(sequence_data).reshape(-1, 1)).flatten()
        
        # Reshape for LSTM input
        sequence_reshaped = sequence_scaled.reshape(1, len(sequence_scaled), 1)
        
        # Make prediction
        prediction_scaled = model.predict(sequence_reshaped, verbose=0)
        
        # Inverse transform to get actual value
        prediction = scaler.inverse_transform(prediction_scaled)[0][0]
        
        return prediction
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def predict_multiple_steps(sequence_data, steps=5, model_path='cpu_predictor_model.keras', scaler_path='cpu_scaler.pkl'):
    """
    Predict multiple future CPU usage values.
    
    Args:
        sequence_data: List or array of the last 12 CPU usage values
        steps: Number of future steps to predict
        model_path: Path to the trained model
        scaler_path: Path to the saved scaler
    
    Returns:
        List of predicted CPU usage values
    """
    try:
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        predictions = []
        current_sequence = list(sequence_data)
        
        for _ in range(steps):
            # Predict next value
            next_pred = predict_cpu_usage(current_sequence[-12:], model_path, scaler_path)
            predictions.append(next_pred)
            
            # Update sequence with prediction for next iteration
            current_sequence.append(next_pred)
        
        return predictions
        
    except Exception as e:
        print(f"Error during multi-step prediction: {e}")
        return None

# This makes the script runnable
if __name__ == "__main__":
    print("Training CPU prediction model...")
    model, scaler, history = train_and_save_model()
    
    if model is not None:
        print("✅ Model training completed successfully!")
        
        # Test prediction with sample data
        sample_sequence = [45.2, 52.3, 48.1, 55.7, 42.8, 38.9, 51.2, 47.6, 49.3, 44.1, 53.8, 46.7]
        prediction = predict_cpu_usage(sample_sequence)
        if prediction:
            print(f"Sample prediction: {prediction:.2f}% CPU usage")
            
        # Multi-step prediction
        multi_pred = predict_multiple_steps(sample_sequence, steps=5)
        if multi_pred:
            print(f"Next 5 predictions: {[f'{p:.2f}%' for p in multi_pred]}")
    else:
        print("❌ Model training failed!")