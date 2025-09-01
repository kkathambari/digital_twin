import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("open_meteo_trichy_hourly_10yr_cleaned.csv")  # replace with your actual filename

# Convert Time to datetime
df['Time'] = pd.to_datetime(df['Time'])

# -------------------------------
# Feature Engineering
# -------------------------------
df['hour'] = df['Time'].dt.hour
df['dayofyear'] = df['Time'].dt.dayofyear
df['month'] = df['Time'].dt.month

# Cyclic encoding for periodicity
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['doy_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
df['doy_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

# Lag features (1 and 2 hours)
df['Direct_Lag1'] = df['Direct Rad (W/m2)'].shift(1)
df['Direct_Lag2'] = df['Direct Rad (W/m2)'].shift(2)
df['Diffuse_Lag1'] = df['Diffuse Rad (W/m2)'].shift(1)
df['Diffuse_Lag2'] = df['Diffuse Rad (W/m2)'].shift(2)

# Drop NA (from lag)
df = df.dropna()

# -------------------------------
# Select Features & Targets
# -------------------------------
features = [
    'Temp (C)', 'Cloudcover (%)', 'Wind Speed (m/s)',
    'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
    'Direct_Lag1', 'Direct_Lag2', 'Diffuse_Lag1', 'Diffuse_Lag2'
]

X = df[features]
y_direct = df['Direct Rad (W/m2)']
y_diffuse = df['Diffuse Rad (W/m2)']

# Train-test split
X_train, X_test, y_train_dir, y_test_dir = train_test_split(X, y_direct, test_size=0.2, shuffle=False)
_, _, y_train_dif, y_test_dif = train_test_split(X, y_diffuse, test_size=0.2, shuffle=False)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# Define GRU Model
# -------------------------------
def build_gru(input_dim):
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=(1, input_dim)),
        Dropout(0.2),
        GRU(64),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# Reshape for GRU
X_train_gru = np.expand_dims(X_train_scaled, axis=1)
X_test_gru = np.expand_dims(X_test_scaled, axis=1)
X_all_gru = np.expand_dims(X_scaled, axis=1)

# -------------------------------
# Training
# -------------------------------
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5)
]

# Train GRU - Direct
gru_direct = build_gru(X_train_scaled.shape[1])
gru_direct.fit(X_train_gru, y_train_dir, validation_data=(X_test_gru, y_test_dir), 
               epochs=50, batch_size=64, callbacks=callbacks, verbose=0)

# Train GRU - Diffuse
gru_diffuse = build_gru(X_train_scaled.shape[1])
gru_diffuse.fit(X_train_gru, y_train_dif, validation_data=(X_test_gru, y_test_dif), 
                epochs=50, batch_size=64, callbacks=callbacks, verbose=0)

# -------------------------------
# Evaluation Function
# -------------------------------
def evaluate(model, X_test, y_test):
    preds = model.predict(np.expand_dims(X_test, axis=1), verbose=0).flatten()
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return mae, rmse, r2

# Evaluate
results = {
    "GRU Direct": evaluate(gru_direct, X_test_scaled, y_test_dir),
    "GRU Diffuse": evaluate(gru_diffuse, X_test_scaled, y_test_dif)
}

# Print results
for k, v in results.items():
    print(f"{k} → MAE: {v[0]:.3f}, RMSE: {v[1]:.3f}, R²: {v[2]:.3f}")

# -------------------------------
# Save Predictions to CSV (FULL DATASET + INPUT FEATURES)
# -------------------------------

# Predict for entire dataset
y_pred_direct_full = gru_direct.predict(X_all_gru, verbose=0).flatten()
y_pred_diffuse_full = gru_diffuse.predict(X_all_gru, verbose=0).flatten()

# Create DataFrame with Datetime, Date, Hour, Features, Actuals, and Predictions
results_df = pd.DataFrame({
    "Datetime": df['Time'].values,
    "Date": df['Time'].dt.date,
    "Hour": df['Time'].dt.hour,
    "Temp (C)": df['Temp (C)'].values,
    "Cloudcover (%)": df['Cloudcover (%)'].values,
    "Wind Speed (m/s)": df['Wind Speed (m/s)'].values,
    "Actual_Direct": y_direct.values,
    "Predicted_Direct": y_pred_direct_full,
    "Actual_Diffuse": y_diffuse.values,
    "Predicted_Diffuse": y_pred_diffuse_full
})

# Save to CSV
results_df.to_csv("gru_predictions_full.csv", index=False)

print("\nPredictions with input features saved to gru_predictions_full.csv")
