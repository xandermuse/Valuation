import DataHandler as dh
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from statsmodels.tsa.arima.model import ARIMA
import prophet as Prophet
import numpy as np
import pandas as pd
import plotnine



class ModelFactory:
    @staticmethod
    def create_model(model_type, data_handler):
        if model_type == 'ARIMA':
            return ArimaModel(data_handler)
        elif model_type == 'LSTM':
            return LstmModel(data_handler)
        elif model_type == 'RandomForest':
            return RandomForestModel(data_handler)
        elif model_type == 'GradientBoosting':
            return GradientBoostingModel(data_handler)
        elif model_type == 'GaussianProcessRegression':
            return GaussianProcessRegressionModel(data_handler)
        else:
            raise ValueError(f"Invalid model type: {model_type}")


class BaseModel:
    def __init__(self, data_handler):
        self.data_handler = data_handler

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def cross_validation(self):
        raise NotImplementedError
    
    def hello(self):
        print("hello from base model")

    def plot(self):
        raise NotImplementedError


class ArimaModel(BaseModel):
    def fit(self, order):
        self.model = ARIMA(self.data_handler.get_data(), order=order)
        self.result = self.model.fit()

    def predict(self, periods):
        forecast = self.result.forecast(steps=periods)
        return forecast

    def cross_validation(self, n_folds):
        data = self.data_handler.get_data()
        n = len(data)
        fold_size = n // n_folds
        df_results = pd.DataFrame(columns=['train_mape', 'test_mape'])

        for i in range(n_folds):
            train = np.concatenate([data[:i*fold_size], data[(i+1)*fold_size:]])
            test = data[i*fold_size:(i+1)*fold_size]
            model = ARIMA(train, order=(1,1,1))
            result = model.fit()
            forecast = result.forecast(steps=len(test))
            mape_train = np.mean(np.abs((train - result.fittedvalues) / train)) * 100
            mape_test = np.mean(np.abs((test - forecast) / test)) * 100
            df_results.loc[i] = [mape_train, mape_test]

        return df_results

    def plot(self, forecast):
        data = self.data_handler.get_data()
        df = pd.DataFrame({'ds': self.data_handler.get_dates(), 'y': data})
        df_forecast = pd.DataFrame({'ds': self.data_handler.get_dates(len(data), len(forecast)), 'y': forecast})
        p = plotnine.ggplot()
        p += plotnine.geom_line(df, plotnine.aes(x='ds', y='y'))
        p += plotnine.geom_line(df_forecast, plotnine.aes(x='ds', y='y'), color='red')
        return p
    
        


class LstmModel(BaseModel):
    def __init__(self, data_handler, look_back=60, epochs=50, batch_size=32):
        super().__init__(data_handler)
        self.look_back = look_back
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def fit(self):
        data = self.data_handler.get_data()
        train_data = data[:-self.look_back]
        test_data = data[-self.look_back:]

        train_data = self.scale_data(train_data)
        X_train, y_train = self.create_dataset(train_data)

        self.model = self.build_model()
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)

    def predict(self):
        data = self.data_handler.get_data()
        test_data = data[-self.look_back:]
        test_data = self.scale_data(test_data)
        X_test, _ = self.create_dataset(test_data)

        predictions = self.model.predict(X_test)
        return self.inverse_scale_data(predictions)

    def cross_validation(self, n_splits=5):
        data = self.data_handler.get_data()
        data = self.scale_data(data)
        X, y = self.create_dataset(data)
        n = len(X)
        fold_size = n // n_splits
        for i in range(n_splits):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size
            val_X = X[val_start:val_end]
            val_y = y[val_start:val_end]
            train_X = np.concatenate([X[:val_start], X[val_end:]], axis=0)
            train_y = np.concatenate([y[:val_start], y[val_end:]], axis=0)
            model = self.build_model()
            model.fit(train_X, train_y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            predictions = model.predict(val_X)
            predictions = self.inverse_scale_data(predictions)
            yield predictions, val_y

    def scale_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def inverse_scale_data(self, data):
        return data * (np.max(self.data_handler.get_data()) - np.min(self.data_handler.get_data())) + np.min(self.data_handler.get_data())

    def create_dataset(self, data):
        dataX, dataY = [], []
        for i in range(len(data) - self.look_back):
            a = data[i:(i + self.look_back)]
            dataX.append(a)
            dataY.append(data[i + self.look_back])
        return np.array(dataX), np.array(dataY)

    def build_model(self):
        model = Sequential()
        model.add(LSTM(50, input_shape=(self.look_back, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def plot(self):
        data = self.data_handler.get_data()
        test_data = data[-self.look_back:]
        test_data = self.scale_data(test_data)
        X_test, _ = self.create_dataset(test_data)

        predictions = self.model.predict(X_test)
        predictions = self.inverse_scale_data(predictions)
        df = pd.DataFrame({'ds': self.data_handler.get_dates(), 'y': data})
        df_forecast = pd.DataFrame({'ds': self.data_handler.get_dates(len(data), len(predictions)), 'y': predictions})
        p = plotnine.ggplot()
        p += plotnine.geom_line(df, plotnine.aes(x='ds', y='y'))
        p += plotnine.geom_line(df_forecast, plotnine.aes(x='ds', y='y'), color='red')
        return p



class RandomForestModel(BaseModel):
    def fit(self):
        # Implement fitting for Random Forest model
        pass

    def predict(self):
        # Implement prediction for Random Forest model
        pass

    def cross_validation(self):
        # Implement cross-validation for Random Forest model
        pass

    def plot(self):
        # Implement plotting for Random Forest model
        pass


class GradientBoostingModel(BaseModel):
    def fit(self):
        # Implement fitting for Gradient Boosting model
        pass

    def predict(self):
        # Implement prediction for Gradient Boosting model
        pass

    def cross_validation(self):
        # Implement cross-validation for Gradient Boosting model
        pass

    def plot(self):
        # Implement plotting for Gradient Boosting model
        pass


class GaussianProcessRegressionModel(BaseModel):
    def fit(self):
        # Implement fitting for Gaussian Process Regression model
        pass

    def predict(self):
        # Implement prediction for Gaussian Process Regression model
        pass

    def cross_validation(self):
        # Implement cross-validation for Gaussian Process Regression model
        pass

    def plot(self):
        # Implement plotting for Gaussian Process Regression model
        pass


# test lstm model
ticker = 'TSLA'
data_handler = dh.DataHandler(ticker)
model = ModelFactory.create_model('LSTM', data_handler)
model.fit()
forecast = model.predict()
pd.DataFrame(forecast).to_csv(f'{ticker}_forecast.csv')
model.plot(forecast)
print(forecast)