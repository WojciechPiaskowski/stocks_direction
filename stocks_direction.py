import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras as tf
import yfinance as yf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam, SGD


pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 40)
sns.set_style('whitegrid')


ticker = 'MSFT'
df = yf.Ticker(ticker).history(start='2000-12-31', end='2020-09-30', interval='5d').iloc[:, :5]

df['moving_average'] = df['Close'].rolling(10).mean()
df = df.dropna()
df['y'] = (df['Close'].diff() > 0).astype(int)
df.reset_index(inplace=True)
df = df.drop(axis=0, labels=1)


df = df.dropna()

plt.plot(df['Close'], label='price')
plt.plot(df['moving_average'], label='moving average')
plt.legend()
plt.title(ticker)
plt.xlabel('time')
plt.ylabel('price')
plt.show()

data = np.array(df.iloc[:, 1:7])
target = np.array(df.iloc[:, 7])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data[:len(data) // 2])
data = scaler.transform(data)

T = 30
D = 6
X = []
Y = []
for t in range(len(data) - T):
    x = data[t:t+T]
    X.append(x)

X = np.array(X).reshape(-1, T, D)
Y = target[T:].reshape(-1, 1)

print(' exogenic values (inputs) shape:', X.shape, '\n', 'endogenic values (outuputs) shape:', Y.shape)

n = len(Y)
X_train = X[:n//2]
X_test = X[n//2:]
Y_train = Y[:n//2]
Y_test = Y[n//2:]

i = Input(shape=(T, D))
x = LSTM(50, return_sequences=True)(i)
x = Dropout(0.3)(x)
x = LSTM(20)(x)
x = Dropout(0.3)(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(i, x)
model.compile(optimizer=SGD(lr=0.1, momentum=0.9), metrics=['accuracy'], loss='binary_crossentropy')
r = model.fit(x=X_train, y=Y_train, epochs=200, shuffle=False, validation_data=(X_test, Y_test))

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()