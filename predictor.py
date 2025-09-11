# predictor.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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
    # 1. Load the realistic usage data
    try:
        df = pd.read_csv('real_usage_data.csv')
        # We will focus on predicting CPU usage for now
        cpu_usage_data = df['cpu_usage'].values.reshape(-1, 1)
    except FileNotFoundError:
        print("Error: 'real_usage_data.csv' not found. Please create it first.")
        return

    # 2. Scale the data to be between 0 and 1
    # Neural networks perform better with scaled data.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(cpu_usage_data)

    # 3. Create training sequences
    # We will use the last 12 data points (1 hour) to predict the next one (5 mins ahead)
    SEQUENCE_LENGTH = 12 
    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)

    # Reshape X to be [samples, time steps, features] which is what LSTM expects
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # 4. Build the LSTM Model
    model = Sequential()
    # Add an LSTM layer with 50 units. input_shape is for the first layer.
    model.add(LSTM(units=50, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)))
    # Add a second LSTM layer
    model.add(LSTM(units=50))
    # Add a dense output layer with 1 neuron to predict the single CPU value
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 5. Train the Model
    print("Training the predictive model...")
    # An epoch is one full pass through the entire training dataset.
    # We'll run it 20 times.
    model.fit(X, y, epochs=20, batch_size=32)
    print("Model training complete.")

    # 6. Save the trained model for later use in our simulation
    model.save('cpu_predictor_model.keras')
    print("Model saved as 'cpu_predictor_model.keras'")
00
# This makes the scri0pt runnable
if __name__ == "__main__":
    train_and_save_model()0