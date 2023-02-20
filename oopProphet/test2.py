
import DataHandler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from plotnine import *
import scipy.stats as stats
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

class LSTMPredictor:
    def __init__(self, lookback, num_epochs, batch_size, lstm_units, dropout_rate, l2_lambda, learning_rate):
        self.lookback = lookback
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.learning_rate = learning_rate
        self.scaler = MinMaxScaler()
        self.model = None

    def prepare_data(self, df):
        data = df.filter(['Close']).values
        data_scaled = self.scaler.fit_transform(data)
        x, y = [], []
        lookback = min(self.lookback, len(data_scaled)-1)
        for i in range(lookback, len(data_scaled)):
            x.append(data_scaled[i - lookback:i, 0])
            y.append(data_scaled[i, 0])
        x, y = np.array(x), np.array(y)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        return x, y


    def build_model(self):
        model = Sequential()
        model.add(LSTM(units=self.lstm_units, return_sequences=True, input_shape=(self.lookback, 1), kernel_regularizer=l2(self.l2_lambda)))
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(units=self.lstm_units, return_sequences=True, kernel_regularizer=l2(self.l2_lambda)))
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(units=self.lstm_units, kernel_regularizer=l2(self.l2_lambda)))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(units=1))
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        self.model = model

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train, epochs=self.num_epochs, batch_size=self.batch_size, verbose=0)

    def predict(self, x_test):
        y_predicted = self.model.predict(x_test)
        y_predicted_scaled = self.scaler.inverse_transform(y_predicted)
        return y_predicted_scaled






if __name__ == '__main__':
    ticker = 'TSLA'
    data = DataHandler.DataHandler(ticker)._get_data()
    print(data)
    data.dropna(inplace=True)
    predictor = LSTMPredictor(lookback=60, num_epochs=100,
                               batch_size=32, lstm_units=50,
                                 dropout_rate=0.2, l2_lambda=0.001,
                                   learning_rate=0.001)
    n = len(data)
    x_train, y_train = predictor.prepare_data(data[:n-60])
    x_test, y_test = predictor.prepare_data(data[n-60:])
    predictor.build_model()
    predictor.train(x_train, y_train)
    y_predicted = predictor.predict(x_test)
    mse = mean_squared_error(y_test, y_predicted)
    print(f'MSE: {mse}')
