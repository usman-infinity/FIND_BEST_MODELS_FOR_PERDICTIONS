import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add

# ----------------------------
# 1️⃣ Load Dataset
# ----------------------------
# Note: Ensure the csv file is in the same directory
df = pd.read_csv("seoul_air_1988_2021.csv")
df = df[['dt','pm10']]
df['dt'] = pd.to_datetime(df['dt'], format='%Y%m%d%H')
df.set_index('dt', inplace=True)
df = df.dropna()

# Using a subset for faster training as per your original script
df = df.head(100000)
print("Dataset shape:", df.shape)

# ----------------------------
# 2️⃣ Normalize Data
# ----------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# ----------------------------
# 3️⃣ Create Time-Series Sequences
# ----------------------------
def create_dataset(data, time_step=24):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        y.append(data[i+time_step, 0])
    return np.array(X), np.array(y)

time_step = 24
X, y = create_dataset(scaled_data, time_step)

# Reshape for [samples, time_steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ----------------------------
# 4️⃣ Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ----------------------------
# 5️⃣ Transformer Encoder Block
# ----------------------------
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(res)

    # Feed Forward Part
    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(inputs.shape[-1])(ff)
    ff = Dropout(dropout)(ff)
    x = Add()([ff, x])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

# ----------------------------
# 6️⃣ Build Improved Transformer Model
# ----------------------------
embed_dim = 64  # Projection dimension
num_heads = 4   # Number of attention heads
ff_dim = 128    # Hidden layer size in feed forward network inside transformer

inputs = Input(shape=(time_step, 1))

# --- Step A: Linear Projection (Embedding) ---
# This turns (24, 1) into (24, 64) so Multi-Head Attention has depth to work with
x = Dense(embed_dim)(inputs)

# --- Step B: Positional Encoding ---
# Adds awareness of time order (Hour 1 vs Hour 24)
positions = tf.range(start=0, limit=time_step, delta=1)
positions = Embedding(input_dim=time_step, output_dim=embed_dim)(positions)
x = x + positions 

# --- Step C: Transformer Blocks ---
x = transformer_encoder(x, head_size=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=0.1)
x = transformer_encoder(x, head_size=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=0.1)

# --- Step D: Final Output Layers ---
x = GlobalAveragePooling1D()(x)
x = Dropout(0.2)(x)
x = Dense(32, activation="relu")(x)
outputs = Dense(1)(x)

model = Model(inputs, outputs)

# ----------------------------
# 7️⃣ Compile Model
# ----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='mean_squared_error',
    metrics=['mae']
)

model.summary()

# ----------------------------
# 8️⃣ Train Model
# ----------------------------
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ----------------------------
# 9️⃣ Predict & Inverse Scale
# ----------------------------
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1,1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))

# ----------------------------
# 🔟 Plot Results
# ----------------------------
plt.figure(figsize=(14,6))
plt.plot(y_test_inv[:200], label='Actual PM10', alpha=0.7)
plt.plot(predictions[:200], label='Predicted PM10', alpha=0.9)
plt.xlabel("Time Steps (Sample of 200)")
plt.ylabel("PM10 Level")
plt.title("Seoul PM10 Forecasting - Enhanced Transformer Model")
plt.legend()
plt.show()

# Plotting Loss
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss (MSE)')
plt.legend()
plt.show()