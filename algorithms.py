import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential, Model
from keras.layers import Dense, GRU, Dropout, BatchNormalization, LSTM, SimpleRNN, Input, Concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

# -------------------------------
# Load and Preprocess Data
# -------------------------------
df = pd.read_csv("open_meteo_trichy_hourly_10yr_cleaned.csv")
df['Time'] = pd.to_datetime(df['Time'])

# Feature Engineering
df['hour'] = df['Time'].dt.hour
df['dayofyear'] = df['Time'].dt.dayofyear
df['month'] = df['Time'].dt.month
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['doy_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
df['doy_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
df['Direct_Lag1'] = df['Direct Rad (W/m2)'].shift(1)
df['Direct_Lag2'] = df['Direct Rad (W/m2)'].shift(2)
df['Diffuse_Lag1'] = df['Diffuse Rad (W/m2)'].shift(1)
df['Diffuse_Lag2'] = df['Diffuse Rad (W/m2)'].shift(2)
df = df.dropna()

features = [
    'Temp (C)', 'Cloudcover (%)', 'Wind Speed (m/s)',
    'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
    'Direct_Lag1', 'Direct_Lag2', 'Diffuse_Lag1', 'Diffuse_Lag2'
]

X = df[features]
y_direct = df['Direct Rad (W/m2)']
y_diffuse = df['Diffuse Rad (W/m2)']

X_train, X_test, y_train_dir, y_test_dir = train_test_split(X, y_direct, test_size=0.2, shuffle=False)
_, _, y_train_dif, y_test_dif = train_test_split(X, y_diffuse, test_size=0.2, shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for RNNs (GRU, LSTM, SimpleRNN)
X_train_rnn = np.expand_dims(X_train_scaled, axis=1)
X_test_rnn = np.expand_dims(X_test_scaled, axis=1)

# -------------------------------
# Define Keras Models
# -------------------------------
def build_gru(input_dim):
    model = Sequential([
        GRU(128, return_sequences=False, input_shape=(1, input_dim)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_lstm(input_dim):
    model = Sequential([
        LSTM(128, return_sequences=False, input_shape=(1, input_dim)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_simplernn(input_dim):
    model = Sequential([
        SimpleRNN(128, return_sequences=False, input_shape=(1, input_dim)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_mlp(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_hybrid_mlp_gru(input_dim):
    # MLP branch
    mlp_input = Input(shape=(input_dim,), name='mlp_input')
    mlp_branch = Dense(128, activation='relu')(mlp_input)
    mlp_branch = BatchNormalization()(mlp_branch)
    mlp_branch = Dense(64, activation='relu')(mlp_branch)
    
    # GRU branch
    gru_input = Input(shape=(1, input_dim), name='gru_input')
    gru_branch = GRU(128, return_sequences=False)(gru_input)
    gru_branch = Dropout(0.2)(gru_branch)
    
    # Concatenate branches
    combined = Concatenate()([mlp_branch, gru_branch])
    
    # Common layers
    z = Dense(64, activation='relu')(combined)
    z = Dense(32, activation='relu')(z)
    output = Dense(1)(z)
    
    model = Model(inputs=[mlp_input, gru_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

# -------------------------------
# Training and Evaluation
# -------------------------------
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5)
]

all_results = {}

print("Training Keras models...")
# Train Keras Models
keras_models = {
    "GRU Direct": build_gru(X_train_scaled.shape[1]),
    "GRU Diffuse": build_gru(X_train_scaled.shape[1]),
    "LSTM Direct": build_lstm(X_train_scaled.shape[1]),
    "LSTM Diffuse": build_lstm(X_train_scaled.shape[1]),
    "SimpleRNN Direct": build_simplernn(X_train_scaled.shape[1]),
    "SimpleRNN Diffuse": build_simplernn(X_train_scaled.shape[1]),
    "MLP Direct": build_mlp(X_train_scaled.shape[1]),
    "MLP Diffuse": build_mlp(X_train_scaled.shape[1]),
    "MLP+GRU Hybrid Direct": build_hybrid_mlp_gru(X_train_scaled.shape[1]),
    "MLP+GRU Hybrid Diffuse": build_hybrid_mlp_gru(X_train_scaled.shape[1])
}

for name, model in keras_models.items():
    if "Hybrid" in name:
        X_train_data = [X_train_scaled, X_train_rnn]
        X_test_data = [X_test_scaled, X_test_rnn]
    elif "MLP" in name:
        X_train_data = X_train_scaled
        X_test_data = X_test_scaled
    else:
        X_train_data = X_train_rnn
        X_test_data = X_test_rnn

    y_train_data = y_train_dir if "Direct" in name else y_train_dif
    y_test_data = y_test_dir if "Direct" in name else y_test_dif

    model.fit(X_train_data, y_train_data, validation_data=(X_test_data, y_test_data), 
              epochs=50, batch_size=64, callbacks=callbacks, verbose=0)
    
    preds = model.predict(X_test_data, verbose=0).flatten()
    mae = mean_absolute_error(y_test_data, preds)
    rmse = np.sqrt(mean_squared_error(y_test_data, preds))
    r2 = r2_score(y_test_data, preds)
    all_results[name] = (mae, rmse, r2)
    print(f"✅ Trained and evaluated {name}")

print("\nTraining Scikit-learn models...")
# Train Scikit-learn Models
skl_models = {
    "Linear Regression Direct": LinearRegression(),
    "Linear Regression Diffuse": LinearRegression(),
    "Decision Tree Direct": DecisionTreeRegressor(),
    "Decision Tree Diffuse": DecisionTreeRegressor(),
    "Random Forest Direct": RandomForestRegressor(n_estimators=100, random_state=42),
    "Random Forest Diffuse": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting Direct": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting Diffuse": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR Direct": SVR(),
    "SVR Diffuse": SVR(),
    "KNN Direct": KNeighborsRegressor(n_neighbors=5),
    "KNN Diffuse": KNeighborsRegressor(n_neighbors=5),
    "MLP (Sklearn) Direct": MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500, early_stopping=True, n_iter_no_change=10, random_state=42),
    "MLP (Sklearn) Diffuse": MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500, early_stopping=True, n_iter_no_change=10, random_state=42)
}

for name, model in skl_models.items():
    y_train_data = y_train_dir if "Direct" in name else y_train_dif
    y_test_data = y_test_dir if "Direct" in name else y_test_dif
    
    model.fit(X_train_scaled, y_train_data)
    
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test_data, preds)
    rmse = np.sqrt(mean_squared_error(y_test_data, preds))
    r2 = r2_score(y_test_data, preds)
    all_results[name] = (mae, rmse, r2)
    print(f"✅ Trained and evaluated {name}")

# -------------------------------
# Print Final Results
# -------------------------------
print("\n" + "="*50)
print("              Final Model Evaluation Results             ")
print("="*50)
for k, v in all_results.items():
    print(f"{k:<30} → MAE: {v[0]:.3f}, RMSE: {v[1]:.3f}, R²: {v[2]:.3f}")
print("="*50)