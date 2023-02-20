import numpy as np
import pandas as pd
import os
import yfinance as yf
from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima_model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# print version of keras and tensorflow
import tensorflow as tf
from tensorflow import keras

class BaseModel(ABC):
    def __init__(self, ticker, train_size):
        self.ticker = ticker
        self.train_size = train_size

    @abstractmethod
    def fit(self, data):
        pass
    
    @abstractmethod
    def predict(self, data):
        pass

    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return {'mse': mse, 'mae': mae}

    def get_train_test_data(self, data):
        n = len(data)
        train_data = data.iloc[:n-self.train_size]
        test_data = data.iloc[n-self.train_size:]
        return train_data, test_data


class ARIMAModel(BaseModel):
    def __init__(self, ticker, train_size, p, d, q):
        super().__init__(ticker, train_size)
        self.p = p
        self.d = d
        self.q = q

    def fit(self, data):
        self.model = ARIMA(data, order=(self.p, self.d, self.q))
        self.model_fit = self.model.fit()

    def predict(self, data):
        return self.model_fit.forecast(steps=self.train_size)


class LSTMModel(BaseModel):
    def __init__(self, ticker, train_size, n_features, n_steps, n_epochs):
        super().__init__(ticker, train_size)
        self.n_features = n_features
        self.n_steps = n_steps
        self.n_epochs = n_epochs

    def fit(self, data):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        data = self.scaler.fit_transform(data)
        X = []
        y = []
        for i in range(self.n_steps, len(data)):
            X.append(data[i-self.n_steps:i])
            y.append(data[i])
        X, y = np.array(X), np.array(y)
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        self.model.add(LSTM(50))
        self.model.add(Dense(1))
        self.model.compile(loss='mse', optimizer='adam')
        self.model.fit(X, y, epochs=self.n_epochs, verbose=0)

    def predict(self, data):
        data = self.scaler.transform(data)
        X = []
        for i in range(self.n_steps, len(data)):
            X.append(data[i-self.n_steps:i])
        X = np.array(X)
        y_pred = self.model.predict(X, verbose=0)
        return self.scaler.inverse_transform(y_pred)


class ModelFactory:
    def __init__(self, ticker, train_size):
        self.ticker = ticker
        self.train_size = train_size

    def get_model(self, model_type, **kwargs):
        if model_type == 'arima':
            return ARIMAModel(self.ticker, self.train_size, **kwargs)
        elif model_type == 'lstm':
            return LSTMModel(self.ticker, self.train_size, **kwargs)