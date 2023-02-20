from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import numpy as np
import pandas as pd
import json
import time
import os
from sklearn.model_selection import ParameterGrid

'''
@Author: Alexander Muse
@Version: 1.0
@Date: 14/2/2023
'''

'''
The ModelHandler class has the following attributes:

    ticker:         The name of the stock ticker.
    current_day:    The current date in the format _%Y_%m_%d.
    data:           The data for the given stock ticker.
    best_params_add:The best hyperparameters for the additive seasonality model.
    best_params_mul:The best hyperparameters for the multiplicative seasonality model.
    best_score_add: The best score for the additive seasonality model.
    best_score_mul: The best score for the multiplicative seasonality model.
    param_grid:     The hyperparameters to be searched for the best model.
    
The class has the following methods:

    fit:            Fits the Prophet model with the given hyperparameters.
    
    predict:        Makes a prediction for the next 30 days using the Prophet model.
    
    cross_validation:  Performs cross-validation on the Prophet model and returns a dataframe.
    
    performance_metrics:  Computes performance metrics for the cross-validated model and returns a dataframe.
    
    plot:           Plots the forecast and saves it as an image in the OutPutImages directory.
    
    write_best_params:   Writes the best hyperparameters for each seasonality mode to a json file.
    
    grid_search:    Searches for the best hyperparameters for each seasonality mode.
    
    run:            Calls the grid_search and fit methods for each seasonality mode, makes a prediction, plots the forecast,
                        writes the best hyperparameters to a json file, and plots the forecast.

To use the class, create an instance of the DataHandler class to obtain the data for the stock ticker, 
then pass the data and ticker to the ModelHandler class. Finally, call the run method to run the model.
    
    ticker = 'TSLA'
    data_handler = DataHandler(ticker)
    data = data_handler._get_data()
    model = ModelHandler(data, ticker)
    model.run()
'''


class ModelHandler:
    def __init__(self, data, ticker, param_grid=None):
        self.data = data  
        self.ticker = ticker
        self.current_day = time.strftime('_%Y_%m_%d')
        self.best_params_add = {}
        self.best_params_mul = {}
        self.best_score_add = np.inf
        self.best_score_mul = np.inf

        self.param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0],
            'holidays_prior_scale': [0.01, 0.1, 1.0],
            'changepoint_range': [0.8, 0.9, 0.95],
            'yearly_seasonality': [True, False],
            'weekly_seasonality': [True, False],
            'daily_seasonality': [True, False],
            'holidays': [None]
        }


    def fit(self, seasonality_mode):
        self.model = Prophet(seasonality_mode=seasonality_mode,
                             changepoint_prior_scale=self.best_params['changepoint_prior_scale'],
                             seasonality_prior_scale=self.best_params['seasonality_prior_scale'],
                             holidays_prior_scale=self.best_params['holidays_prior_scale'],
                             changepoint_range=self.best_params['changepoint_range'],
                             yearly_seasonality=self.best_params['yearly_seasonality'],
                             weekly_seasonality=self.best_params['weekly_seasonality'],
                             daily_seasonality=self.best_params['daily_seasonality'],
                             holidays=self.best_params['holidays'])
        self.model.fit(self.data)

    def predict(self, periods):
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        return forecast
    
    def cross_validation(self):
        df_cv = cross_validation(self.model, horizon='30 days', parallel="processes")
        return df_cv
    
    def performance_metrics(self, df_cv):
        df_p = performance_metrics(df_cv)
        return df_p
    
    def save_model(self, filename):
        self.model.save(filename)
    
    def plot(self, forecast , prefix='TEST'):
        plot = self.model.plot(forecast)
        if not os.path.exists(f'oopProphet/OutPutImages/{self.ticker}'):
            os.makedirs(f'oopProphet/OutPutImages/{self.ticker}')
        plot.savefig(f'oopProphet/OutPutImages/{prefix}_{self.ticker}{self.current_day}.png')
    
    def write_best_params(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.best_params, f)

    def grid_search(self, seasonality_mode):
        best_params = {}
        best_score = np.inf
        start_time = time.time()
        n_models = len(list(ParameterGrid(self.param_grid)))
        
        for i, params in enumerate(ParameterGrid(self.param_grid)):
            model = Prophet(seasonality_mode=seasonality_mode,
                            changepoint_prior_scale=params['changepoint_prior_scale'],
                            seasonality_prior_scale=params['seasonality_prior_scale'],
                            holidays_prior_scale=params['holidays_prior_scale'],
                            changepoint_range=params['changepoint_range'],
                            yearly_seasonality=params['yearly_seasonality'],
                            weekly_seasonality=params['weekly_seasonality'],
                            daily_seasonality=params['daily_seasonality'],
                            holidays=params['holidays'])

            if (i + 1) % 1 == 0:
                elapsed_time = time.time() - start_time
                remaining_time = (elapsed_time / (i + 1)) * (n_models - i - 1)
                print(f"\n\nEstimated remaining time: {remaining_time / 60:.2f} minutes\n\n")
                time.sleep(5)

            model.fit(self.data)
            df_cv = cross_validation(model, horizon='30 days', parallel="processes")
            df_p = performance_metrics(df_cv)
            
            if df_p['mape'].values[0] < best_score:
                best_score = df_p['mape'].values[0]
                best_params = params
                print(f"New best score: {best_score}")
                print(f"New best params: {best_params}")
        
        if seasonality_mode == 'additive':
            self.best_params_add = best_params
            self.best_score_add = best_score
        elif seasonality_mode == 'multiplicative':
            self.best_params_mul = best_params
            self.best_score_mul = best_score



    def run(self):
            print(f'n_models: {len (list(ParameterGrid(self.param_grid)))}')
            time.sleep(3)
            seasonality_List = ['additive', 'multiplicative']
            for seasonality_mode in seasonality_List:
                self.grid_search(seasonality_mode)
                self.write_best_params(f'oopProphet/BestParams/best_params_{seasonality_mode}_{self.ticker}.json')
                self.fit(seasonality_mode)
                future = self.model.make_future_dataframe(periods=30)
                forecast = self.model.predict(future)
                self.plot(forecast, prefix=seasonality_mode)
                print(forecast)
                print(self.best_params)   

