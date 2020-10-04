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


ticker = 'NVDA'
df = yf.Ticker(ticker).history(start='2010-12-31', end='2020-09-30', interval='5d').iloc[:, :5]

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

X = np.array(df.iloc[:, 1:7])
Y = np.array(df.iloc[:, 7])

X.shape
Y.shape

n = len(Y)
X_train = X[:n//2]
X_test = X[n//2:]
Y_train = X[:n//2]
Y_test = X[n//2:]


i = Input()
x = LSTM(10)(i)
x = LSTM(5)(x)
x = Dense(activation='sigmoid')(x)
model = Model(i, x)
model.compile(optimizer=Adam(lr=0.1), metrics=['accuracy'], loss='binary_crossentropy')

