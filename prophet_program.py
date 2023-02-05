import numpy as np
import pandas as pd
from prophet import Prophet
import datetime as dt
import yfinance as yf
import os
import warnings
import logging

from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly, plot_components_plotly


class ProphetModel:
    def __init__(self, ticker):
        self.ticker = ticker
        self.today = dt.datetime.today().strftime('_%Y_%m_%d')
        try :
            self.data = pd.read_csv(f'./data/{self.ticker}{self.today}.csv')
            print(f"Successfully read data for {self.ticker} from csv")
        except Exception as e:
            print(f"Failed to read data for {self.ticker} from csv: {e}")
            self.data = self.write_data()


        self.model_add = Prophet(seasonality_mode='additive', 
                                daily_seasonality=True,         # daily seasonality
                                weekly_seasonality=True,        # weekly seasonality
                                yearly_seasonality=True,        # yearly seasonality
                                changepoint_prior_scale=0.2,    # strength of the trend changepoints
                                changepoint_range=0.95,         # proportion of history in which trend changepoints will be estimated
                                seasonality_prior_scale=1.0,   # strength of the seasonality model
                                holidays_prior_scale=0.01,
                                interval_width=0.95,            # width of the uncertainty intervals
                                uncertainty_samples=10000       # number of simulated draws used to estimate uncertainty intervals
                                )

        self.model_mult = Prophet(seasonality_mode='multiplicative',
                                daily_seasonality=True,
                                weekly_seasonality=True, 
                                yearly_seasonality=True, 
                                changepoint_prior_scale=0.2, 
                                changepoint_range=0.9, 
                                seasonality_prior_scale=1.0,
                                holidays_prior_scale=0.01,
                                interval_width=0.95,
                                uncertainty_samples=10000
                                )


        self.param_grid = {
            'seasonality_mode': ['additive'],
            'daily_seasonality': [True],
            'weekly_seasonality': [True],
            'yearly_seasonality': [True],
            'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.2],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'changepoint_range': [0.8, 0.9, 0.95]
        }
        
    def get_data(self):
        data = yf.download(self.ticker, period='8y', interval='1d')
        data = data.reset_index()
        data = data.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
        df = pd.DataFrame(data)
        return df

    def write_data(self):
        # try and made data directory if it doesn't exist
        try:
            os.mkdir('./data')
            print("Made data directory")
        except FileExistsError:
            pass
        try:
            df = self.get_data()
            df.to_csv(f'./data/{self.ticker}{self.today}.csv', index=False)
            print(f"Successfully wrote data for {self.ticker} to csv")
        except Exception as e:
            print(f"Failed to write data for {self.ticker} to csv: {e}")

    def forecast_add(self):
        df = self.get_data()
        self.model_add.fit(df)
        future = self.model_add.make_future_dataframe(periods=30)
        forecast_add = self.model_add.predict(future)
        y_true = df["y"].values
        y_pred = forecast_add["yhat"].values[:-30]
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{self.ticker} {self.today} add MAE: {mae}")
        print(f"{self.ticker} {self.today} add MSE: {mse}")
        print(f"{self.ticker} {self.today} add R2: {r2}")

        forecast_df = forecast_add[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        actuals_df = df[['ds', 'y']].copy()
        merged_df = pd.merge(actuals_df, forecast_df, on='ds', how='outer')
        merged_df['yhat'] = merged_df['yhat'].fillna(merged_df['y'])
        merged_df['yhat_lower'] = merged_df['yhat_lower'].fillna(merged_df['y'])
        merged_df['yhat_upper'] = merged_df['yhat_upper'].fillna(merged_df['y'])
        merged_df = merged_df.set_index('ds')
        return forecast_add , merged_df


    def forecast_mult(self):
        df = self.get_data()
        self.model_mult.fit(df)
        future = self.model_mult.make_future_dataframe(periods=30)
        forecast_mult = self.model_mult.predict(future)
        y_true = df["y"].values
        y_pred = forecast_mult["yhat"].values[:-30]
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{self.ticker} {self.today} mult MAE: {mae}")
        print(f"{self.ticker} {self.today} mult MSE: {mse}")
        print(f"{self.ticker} {self.today} mult R2: {r2}")
        return forecast_mult


    def plot_components(self, model, forecast, prefix):
        x = model.plot_components(forecast)    
        x.savefig(f'output_images/PROPHET/{self.ticker}/{self.ticker}_{prefix}_plot_components{self.today}.jpg')
        # return x

        
    def plot_forecast(self, model, forecast, prefix):
        x = model.plot(forecast)
        if os.path.exists(f'output_images/PROPHET/{self.ticker}'):
            x.savefig(f'output_images/PROPHET/{self.ticker}/{self.ticker}_{prefix}_plot_forecast{self.today}.jpg')
        else:
            os.mkdir(f'output_images/PROPHET/{self.ticker}')
            x.savefig(f'output_images/PROPHET/{self.ticker}/{self.ticker}_{prefix}_plot_forecast{self.today}.jpg')
        # return x   

    
    def cross_validate(self):
        cv_results_add = cross_validation(self.model_add, initial='365 days', period='30 days', horizon = '7 days', parallel='processes')
        cv_results_mult = cross_validation(self.model_mult, initial='365 days', period='30 days', horizon = '7 days', parallel='processes')
        return cv_results_add, cv_results_mult  

    def hyperparameter_tuning(self):
        df = self.get_data()
        param_grid = ParameterGrid(self.param_grid)
        results = []
        for params in param_grid:
            model = Prophet(**params)
            model.fit(df)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            y_true = df["y"].values
            y_pred = forecast["yhat"].values[:-30]
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            results.append((params, mae, mse, r2))
        return results




def run(tickerList):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    tickerList = tickerList
    print(tickerList)
    today = dt.date.today().strftime('%Y-%m-%d')
    print(today)
    running = True
    for ticker in tickerList:
        # try to destroy the old prophet model
        try:
            del prophet
        except: 
            pass

        try:
            os.mkdir(f'output_images/PROPHET/{ticker}')
        except:
            print(f'output_images/PROPHET/{ticker} already exists')


        if os.path.exists(f'output_images/PROPHET/{ticker}/{ticker}_plot_forecast{today}.jpg'):
            print(f'output_images/PROPHET/{ticker}/{ticker}_plot_forecast{today}.jpg already exists')
            continue
        
        else:
            print(f'DOES NOT EXIST : output_images/PROPHET/{ticker}/{ticker}_plot_forecast{today}.jpg')
            try:
                os.mkdir(f'output_data/PROPHET/{ticker}')
            except:
                print(f'{ticker} already exists')           


        if os.path.exists(f'output_data/PROPHET/{ticker}/{ticker}_forecast_mult{today}.jpg'):
            print(f'output_data/PROPHET/{ticker}/{ticker}_forecast_mult{today}.jpg already exists')
            continue
        else:
            print('-----------------------')
            print('Running Prophet Model...')
            print(f'Creating {ticker} Model...')
            prophet = ProphetModel(ticker)
            print(f'{ticker} Model created')
            prophet.write_data()
            forecast_add, merged_df = prophet.forecast_add()
            print(f'{ticker} forecast_add created')
            forecast_mult = prophet.forecast_mult()
            print(f'{ticker} forecast_mult created')
            prophet.plot_forecast(prophet.model_add, forecast_add, 'add')
            print(f'{ticker} plot_forecast_add created')
            prophet.plot_forecast(prophet.model_mult, forecast_mult, 'mult')
            print(f'{ticker} plot_forecast_mult created')
            print(f'{ticker} created')
            print('-----------------------')


if __name__ == '__main__':
    tickerList = ['AAPL', 'AMD', 'VALE', 'AMZN', 'F', 'GM', 'TM', 'TSLA', 'GOOGL']
    tickerList.sort()
    run(tickerList)