import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow.keras.backend as K
import math
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r'D:\Manoj_Honors\Imputed_Predictions(Univariate,Bivariate).csv')

# Drop Null values
data = data.dropna()

# Drop 'Unnamed: 0' as it's just an index
data_dropped = data.drop(columns=['Unnamed: 0'])

# Convert 'Time' to datetime format and extract useful time features
data_dropped['Time'] = pd.to_datetime(data_dropped['Time'])
data_dropped['Hour'] = data_dropped['Time'].dt.hour
data_dropped['Day'] = data_dropped['Time'].dt.day
data_dropped['Month'] = data_dropped['Time'].dt.month

# Positional encoding for Latitude and Longitude
data_dropped['Lat_sin'] = np.sin(np.radians(data_dropped['Lat']))
data_dropped['Lat_cos'] = np.cos(np.radians(data_dropped['Lat']))
data_dropped['Lon_sin'] = np.sin(np.radians(data_dropped['Lon']))
data_dropped['Lon_cos'] = np.cos(np.radians(data_dropped['Lon']))

# Drop the 'Time', 'Lat', and 'Lon' columns
data_dropped = data_dropped.drop(columns=['Time', 'Lat', 'Lon'])

# Separate input features and target features
input_features = data_dropped.drop(columns=['NO2', 'PM2.5', 'Ozone'])
target_features = data_dropped[['NO2', 'PM2.5', 'Ozone']]

# Normalize input features and target features separately
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(input_features)
y_scaled = scaler_y.fit_transform(target_features)

# Combine scaled inputs and targets back for sequence splitting
data_scaled = np.hstack((X_scaled, y_scaled))

# Function to create sliding windows
def split_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix >= len(data):
            break
        seq_x, seq_y = data[i:end_ix, :], data[end_ix, -3:]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Set the number of time steps
n_steps = 4

# Split into input (X) and output (y)
X, y = split_sequences(data_scaled, n_steps)

# Split into train and validation sets (80%-20%)
train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Custom R² metric
def r2_keras(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())

# Define the transformer block
def transformer_encoder(inputs, num_heads, ff_dim, dropout_rate):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = LayerNormalization(epsilon=1e-6)(attention_output + ff_output)

    return ff_output

# Define the transformer model
def build_transformer_model(input_shape, num_heads, ff_dim, num_layers, dropout_rate):
    inputs = Input(shape=input_shape)
    x = inputs

    # Apply transformer layers
    for _ in range(num_layers):
        x = transformer_encoder(x, num_heads, ff_dim, dropout_rate)

    # Global average pooling to reduce the sequence dimension
    x = GlobalAveragePooling1D()(x)

    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)

    # Output layer (3 pollutants)
    outputs = Dense(3)(x)
    model = Model(inputs, outputs)

    return model

# Build the model
input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
transformer_model = build_transformer_model(
    input_shape=input_shape,
    num_heads=8,
    ff_dim=128,
    num_layers=4,
    dropout_rate=0.3
)

# Compile the model
transformer_model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae', r2_keras])

# Callbacks for regularization
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

# Train the transformer model
history_transformer = transformer_model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)

# Evaluate the model
y_pred_transformer = transformer_model.predict(X_val)

# Rescale predictions
y_pred_transformer_rescaled = scaler_y.inverse_transform(y_pred_transformer)
y_val_rescaled = scaler_y.inverse_transform(y_val)

# Function to calculate metrics for each pollutant
def calculate_metrics(y_true, y_pred, pollutant_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"Metrics for {pollutant_name}:")
    print(f"MSE: {mse:.9f}")
    print(f"MAE: {mae:.9f}")
    print(f"RMSE: {rmse:.9f}")
    print(f"R²: {r2:.9f}")
    print(f"MAPE: {mape:.9f}%")
    print("-" * 40)

# Calculate metrics for each pollutant
calculate_metrics(y_val_rescaled[:, 0], y_pred_transformer_rescaled[:, 0], "NO2")
calculate_metrics(y_val_rescaled[:, 1], y_pred_transformer_rescaled[:, 1], "PM2.5")
calculate_metrics(y_val_rescaled[:, 2], y_pred_transformer_rescaled[:, 2], "Ozone")

# Plot training and validation R²
plt.plot(history_transformer.history['r2_keras'], label='Train R²')
plt.plot(history_transformer.history['val_r2_keras'], label='Val R²')
plt.legend()
plt.show()

# Save the transformer model
transformer_model.save('optimized_pollutant_transformer_model.keras')
print("Transformer model saved as 'optimized_pollutant_transformer_model.keras'.")
