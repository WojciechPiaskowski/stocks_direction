# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, SimpleRNN, GRU
from tensorflow.keras.optimizers import Adam, SGD

# style settings
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 40)
sns.set_style('whitegrid')

# select a ticker to download data from yahoo finance
ticker = 'BAC'
df = yf.Ticker(ticker).history(start='1970-12-31', end='2020-09-30', interval='5d').iloc[:, :5]

# add a moving average and target columns, drop first row because we cannot know if direction was increase or decrease
df['moving_average'] = df['Close'].rolling(10).mean()
df['y'] = (df['Close'].diff() > 0).astype(int)
df.reset_index(inplace=True)
df = df.drop(axis=0, labels=1)

# plot the price vs time
plt.plot(df['Close'], label='price')
plt.plot(df['moving_average'], label='moving average')
plt.legend()
plt.title(ticker)
plt.xlabel('time')
plt.ylabel('price')
plt.show()

# the process is non-stationary, so replace the data with first differences, but keep the same target (and drop NA rows)
df2 = df
df = df.diff()
df['y'] = df2['y']
df = df.dropna()

# cast data into numpy arrays
data = np.array(df.iloc[:, 1:7])
target = np.array(df.iloc[:, 7])

# scale the data using standard scaler ((X - mi)/sd)
from sklearn.preprocessing import StandardScaler, Normalizer
scaler = StandardScaler()
scaler.fit(data[:len(data) // 2])
data = scaler.transform(data)

T = 100 # number of lag inputs used
D = 6 # number of dimensions/features
X = []
Y = []

# append data in X array
for t in range(len(data) - T):
    x = data[t:t+T]
    X.append(x)

# reshape the data to fit LSTM network
X = np.array(X).reshape(-1, T, D)
Y = target[T:].reshape(-1, 1)

print(' exogenic values (inputs) shape:', X.shape, '\n', 'endogenic values (outuputs) shape:', Y.shape)

# divide data in training and testing samples (without shuffling it because its a time series)
n = len(Y)
X_train = X[:n//2]
X_test = X[n//2:]
Y_train = Y[:n//2]
Y_test = Y[n//2:]

# create a network
i = Input(shape=(T, D))
x = LSTM(50, return_sequences=True)(i)
x = Dropout(0.3)(x)
x = LSTM(20)(x)
x = Dropout(0.3)(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(i, x)
model.compile(optimizer=Adam(lr=0.1), metrics=['accuracy'], loss='binary_crossentropy')
r = model.fit(x=X_train, y=Y_train, epochs=50, validation_data=(X_test, Y_test), shuffle=False)

# plot loss and validation loss
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

# Unfortunately the model does not give good results even after trying many different variants:
# original data, first and second differences, diffrent numbers of LSTM units, layers, dropout layers, no. of epochs,
# LSTM, GRU, SimpleRNN units

# it might be impossible to get better results at predicting direction of the stock movement
# things to further try:
# - ARIMA / GARCH models analysis --- there is conditional heteroscedacity in first diffrences, so GARCH might be
# working better, or at least give some insight if we analyze statistically significant lag periods

# another thing to try is to use more metrics as exgoenic variables -- eg. EPS, P/B, FCF
