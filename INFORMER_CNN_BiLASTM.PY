import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

# ----------------------------
# 1️⃣ Load Dataset
# ----------------------------
df = pd.read_csv("seoul_air_1988_2021.csv")

# Print column names to check dataset
print("Dataset Columns:", df.columns)

# Rename possible PM2.5 column names
if 'pm2_5' in df.columns:
    df.rename(columns={'pm2_5':'pm25'}, inplace=True)
elif 'pm2.5' in df.columns:
    df.rename(columns={'pm2.5':'pm25'}, inplace=True)
elif 'PM2.5' in df.columns:
    df.rename(columns={'PM2.5':'pm25'}, inplace=True)
elif 'PM25' in df.columns:
    df.rename(columns={'PM25':'pm25'}, inplace=True)

# Select required columns
df = df[['dt','pm10','pm25']]

# ----------------------------
# Convert datetime
# ----------------------------
df['dt'] = pd.to_datetime(df['dt'], format='%Y%m%d%H')
df.set_index('dt', inplace=True)

# Remove missing values
df = df.dropna()

# Optional: use smaller dataset for testing
df = df.head(100000)

print("Dataset shape:", df.shape)

# ----------------------------
# 2️⃣ Normalize Data
# ----------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# ----------------------------
# 3️⃣ Create Time-Series Dataset
# ----------------------------
def create_dataset(data, time_step=24):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), :])
        y.append(data[i+time_step, :])
    return np.array(X), np.array(y)

time_step = 24
X, y = create_dataset(scaled_data, time_step)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ----------------------------
# 4️⃣ Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ----------------------------
# 5️⃣ Positional Encoding
# ----------------------------
class PositionalEncoding(layers.Layer):
    def __init__(self, time_step, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(time_step, d_model)

    def get_angles(self, pos, i, d_model):
        angles = 1 / np.power(10000, (2*(i//2))/np.float32(d_model))
        return pos * angles

    def positional_encoding(self, time_step, d_model):
        angle_rads = self.get_angles(
            np.arange(time_step)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )

        angle_rads[:,0::2] = np.sin(angle_rads[:,0::2])
        angle_rads[:,1::2] = np.cos(angle_rads[:,1::2])

        return tf.cast(angle_rads[np.newaxis,...], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# ----------------------------
# 6️⃣ CNN + Informer + BiLSTM Model
# ----------------------------
d_model = 64

inputs = layers.Input(shape=(time_step,2))

x = layers.Conv1D(64,3,activation='relu')(inputs)
x = layers.MaxPooling1D(2)(x)

x = layers.Dense(d_model)(x)

x = PositionalEncoding(time_step//2, d_model)(x)

attention = layers.MultiHeadAttention(
    num_heads=4,
    key_dim=d_model
)(x,x)

x = layers.Add()([x,attention])
x = layers.LayerNormalization()(x)

x = layers.Bidirectional(layers.LSTM(64))(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(2)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# ----------------------------
# 7️⃣ Train
# ----------------------------
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test,y_test)
)

# ----------------------------
# 8️⃣ Predict
# ----------------------------
predictions = model.predict(X_test)

predictions_inv = scaler.inverse_transform(predictions)
y_test_inv = scaler.inverse_transform(y_test)

# ----------------------------
# 9️⃣ Plot PM10
# ----------------------------
plt.figure(figsize=(12,5))
plt.plot(y_test_inv[:,0],label='Actual PM10')
plt.plot(predictions_inv[:,0],label='Predicted PM10')
plt.title("PM10 Prediction")
plt.legend()
plt.show()

# ----------------------------
# 🔟 Plot PM2.5
# ----------------------------
plt.figure(figsize=(12,5))
plt.plot(y_test_inv[:,1],label='Actual PM2.5')
plt.plot(predictions_inv[:,1],label='Predicted PM2.5')
plt.title("PM2.5 Prediction")
plt.legend()
plt.show()