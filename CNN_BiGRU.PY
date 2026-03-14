import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Bidirectional, GRU
from tensorflow.keras.layers import Dense, Dropout

# ----------------------------
# 1️⃣ Load Dataset
# ----------------------------
df = pd.read_csv("seoul_air_1988_2021.csv")

# Only use relevant columns
df = df[['dt','pm10']]

# Convert dt column to datetime
df['dt'] = pd.to_datetime(df['dt'], format='%Y%m%d%H')

# Set datetime as index
df.set_index('dt', inplace=True)

# Drop missing values
df = df.dropna()

print("Dataset shape:", df.shape)

# Use only first 100,000 rows for quick testing
df = df.head(100000)

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

# Reshape for CNN input
X = X.reshape(X.shape[0], X.shape[1], 1)

print("X shape:", X.shape, "y shape:", y.shape)

# ----------------------------
# 4️⃣ Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ----------------------------
# 5️⃣ Build CNN-BiGRU Model
# ----------------------------
model = Sequential()

# CNN layer
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(time_step,1)))
model.add(MaxPooling1D(pool_size=2))

# BiGRU layer
model.add(Bidirectional(GRU(64)))

# Dropout
model.add(Dropout(0.2))

# Output layer
model.add(Dense(1))

# ----------------------------
# 6️⃣ Compile Model
# ----------------------------
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# ----------------------------
# 7️⃣ Train Model
# ----------------------------
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ----------------------------
# 8️⃣ Predict
# ----------------------------
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1,1))

# Inverse transform y_test
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))

# ----------------------------
# 9️⃣ Plot Results
# ----------------------------
plt.figure(figsize=(12,5))
plt.plot(y_test_inv, label='Actual PM10')
plt.plot(predictions, label='Predicted PM10')
plt.xlabel("Time Steps")
plt.ylabel("PM10")
plt.title("Seoul PM10 Forecasting - CNN-BiGRU")
plt.legend()
plt.show()