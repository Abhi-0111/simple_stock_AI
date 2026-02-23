import kagglehub
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. Get Real Data
path = kagglehub.dataset_download("camnugent/sandp500")
df = pd.read_csv(os.path.join(path, "all_stocks_5yr.csv"))
stock = df[df['Name'] == 'GOOGL'].copy() # Let's predict Google
prices = stock['close'].values.reshape(-1, 1)

# 2. Preprocessing (Scale data between 0 and 1)
# Neural networks work much better with small numbers!
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices)

# 3. Create "Windows" of data (Use last 5 days to predict the next)
X, y = [], []
for i in range(5, len(scaled_prices)):
    X.append(scaled_prices[i-5:i, 0]) # Input: Last 5 days
    y.append(scaled_prices[i, 0])     # Target: Today's price

X, y = np.array(X), np.array(y)

# Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Build the TensorFlow Model
model = Sequential([
    Dense(10, activation='relu', input_shape=(5,)), # 10 "neurons" looking for patterns
    Dense(5, activation='relu'),                    # 5 more neurons to refine the guess
    Dense(1)                                        # 1 final price prediction
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. Train the Model
print("Training the neural network...")
model.fit(X_train, y_train, epochs=20, batch_size=8, verbose=0)

# 6. Predict and Visualize
predictions = model.predict(X_test)

# Un-scale the data to see real dollar amounts
predictions = scaler.inverse_transform(predictions)
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(10,5))
plt.plot(y_test_real, label='Actual Google Price')
plt.plot(predictions, label='AI Prediction', linestyle='--')
plt.legend()
plt.title('Google Stock Prediction')
plt.show()