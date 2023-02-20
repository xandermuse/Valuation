import yfinance as yf
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import datetime as dt

class StockForecast:
    """
    Class for creating and optimizing a stock forecast using the Prophet library.
    """
    def __init__(self, ticker, model_type='multiplicative', start_date='2015-01-01', end_date='2022-01-01'):
        """
        Initializes the StockForecast class.
        
        :param ticker: str, the ticker symbol of the stock
        :param model_type: str, the type of Prophet model to use (default='multiplicative')
        :param start_date: str, start date for retrieving stock data (default='2015-01-01')
        :param end_date: str, end date for retrieving stock data (default='2022-01-01')
        """
        self.ticker = ticker
        self.model_type = model_type
        self.start_date = start_date
        self.end_date = end_date
        self.df = None
        self.best_params = None
        self.best_result = None
        self.itterlist = []

    def get_data(self):
        """
        Retrieves stock data for the specified ticker and date range.
        """
        self.df = pd.DataFrame(yf.download(self.ticker, start=self.start_date, end=self.end_date))
        self.df = self.df[['Close']]
        self.df.reset_index(inplace=True)
        self.df.columns = ['ds', 'y']
        print(self.df)

    def prophet_model(self, params=None):
        """
        Creates a Prophet model with specified parameters.
        
        :param params: dict, the parameters for the Prophet model (default=None)
        :return: pd.DataFrame, the forecast results
        """
        model = Prophet(growth = "linear")
        if params:
            model.seasonality_prior_scale = params['seasonality_prior_scale']
            model.holidays_prior_scale = params['holidays_prior_scale']
            model.changepoint_prior_scale = params['changepoint_prior_scale']
        model.fit(self.df)
        future = model.make_future_dataframe(periods=90)
        forecast = model.predict(future)
        return forecast
    

    def optimize_parameters(self, num_iter=100, init_temp=100, cooling_rate=0.95):
        """
        Optimizes the Prophet model using the simulated annealing algorithm.
        
        :param num_iter: int, number of iterations for the optimization (default=100)
        :param init_temp: int, initial temperature for the optimization (default=100)
        :param cooling_rate: float, rate of cooling for the optimization (default=0.95)
        """
        current_params = {'seasonality_prior_scale': 1.0, 'holidays_prior_scale': 1.0, 'changepoint_prior_scale': 0.05}
        best_params = current_params
        temperature = init_temp
        best_result = self.prophet_model(current_params)['yhat'].mean()
        for i in range(num_iter):
            next_params = {
                'seasonality_prior_scale': current_params['seasonality_prior_scale'] + random.uniform(-0.1, 0.1),
                'holidays_prior_scale': current_params['holidays_prior_scale'] + random.uniform(-0.1, 0.1),
                'changepoint_prior_scale': current_params['changepoint_prior_scale'] + random.uniform(-0.1, 0.1)
            }
            next_result = self.prophet_model(next_params)['yhat'].mean()
            delta_e = next_result - best_result
            if delta_e > 0:
                current_params = next_params
                best_params = next_params
                best_result = next_result
                self.clear()
                print(f'\n\nIteration: {i} best_result: {best_result}\n\n')
                self.itterlist.append([i,best_result])
                print(self.itterlist)
            else:
                p = math.exp(delta_e / temperature)
                if random.uniform(0, 1) < p:
                    current_params = next_params
                params = next_params
                temperature *= cooling_rate
            self.best_params = best_params
            self.best_result = self.prophet_model(best_params)

    def plot_result(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.df['ds'], self.df['y'], label='actual')
        plt.plot(self.best_result['ds'], self.best_result['yhat'], label='forecast')
        plt.legend()
        plt.show()

    def save_result(self, file_name):
        self.best_result.to_csv(file_name, index=False)

    # clear console
    def clear(self):
        print("\033[H\033[J")



if __name__ == '__main__':
    ticker = 'TSLA'
    model_type = 'multiplicative'
    start_date = '2020-01-01'
    end_date = '2022-01-01'


    sf = StockForecast(ticker, model_type, start_date, end_date)
    data = sf.get_data()
    start = dt.datetime.now()
    sf.optimize_parameters()
    end = dt.datetime.now()
    sf.plot_result()
    sf.save_result('best_result.csv')
    sf.clear()
    print(sf.best_params)
    for i in sf.itterlist:
        print(f'Iteration: {i[0]} best_result: {i[1]}')
    print(f'Time taken: {end - start}')