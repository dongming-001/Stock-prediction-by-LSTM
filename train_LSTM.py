

def train_model(df):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam

    # Load the training data
    # df = pd.read_csv('large_example_stock_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))

    # Create the training and testing data, labels
    def create_dataset(dataset, time_step=60):
        X, Y = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   
            X.append(a)
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    # Split data into train and test
    training_size = int(len(scaled_data) * 0.8)
    test_size = len(scaled_data) - training_size
    train_data, test_data = scaled_data[0:training_size,:], scaled_data[training_size:len(scaled_data),:1]

    # Reshape into X=t, t+1, t+2,...,t+59 and Y=t+60
    time_step = 60
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

    # Build the LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test))

    # Assuming the model was trained on sequences of 60 days
    # Let's generate a "recent_data" sequence from the last part of our generated dataset

    # Number of days the model was trained on
    time_step = 60

    # Extract the last `time_step` days from the `large_stock_data` as the recent data
    # Ensure to use the 'Close' prices and scale them as per the model's training process

    # Simulating the scaling process (in real scenarios, use the same MinMaxScaler instance used during training)
    from sklearn.preprocessing import MinMaxScaler
    large_stock_data = df

    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit the scaler to the entire dataset's Close prices to simulate a realistic scenario
    scaler.fit(large_stock_data['Close'].values.reshape(-1, 1))

    # Extract the last `time_step` days and scale
    recent_data_scaled = scaler.transform(large_stock_data['Close'].values[-time_step:].reshape(-1, 1))

    # Reshape for the model (model expects [samples, time steps, features])
    recent_data_shaped = recent_data_scaled.reshape((1, time_step, 1))

    recent_data_shaped.shape, recent_data_scaled[-1]  # Show shape and the last scaled value

    # After obtaining each predicted_price from the model
    predict_days = 5
    for _ in range(5):  # Predict the next 5 days
        predicted_price = model.predict(recent_data_shaped)
        
        # Ensure the predicted_price has the correct shape for appending
        # Reshape predicted_price to have a shape of (1, 1) before appending
        predicted_price_shaped = predicted_price.reshape(1, 1, 1)  # Correct shape to append
        
        # Append the new prediction, ensuring dimensions match
        recent_data_shaped = np.append(recent_data_shaped[:, 1:, :], predicted_price_shaped, axis=1)

    predicted_prices_scaled = recent_data_shaped[:, -predict_days:, 0]

    # Inverse transform the predictions to get them back to the original scale
    # First, reshape to (-1, 1) because inverse_transform expects a 2D array
    predicted_prices_scaled_reshaped = predicted_prices_scaled.reshape(-1, 1)
    predicted_prices_unscaled = scaler.inverse_transform(predicted_prices_scaled_reshaped)
    # print(np.shape(predicted_prices_unscaled))
    return predicted_prices_unscaled[:,0]