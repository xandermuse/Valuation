import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from plotnine import *

class ErrorAnalyzer:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
    
    def calculate_metrics(self):
        """
        Calculates mean squared error (MSE) and mean absolute error (MAE) for the predicted and true values.
        Returns a dictionary with the calculated metrics.
        """
        mse = mean_squared_error(self.y_true, self.y_pred)
        mae = mean_absolute_error(self.y_true, self.y_pred)
        return {'MSE': mse, 'MAE': mae}
    
    def generate_error_df(self):
        """
        Generates a Pandas DataFrame containing the predicted and true values, as well as the prediction errors.
        Returns the DataFrame.
        """
        errors = self.y_true - self.y_pred
        df = pd.DataFrame({'y_true': self.y_true, 'y_pred': self.y_pred, 'error': errors})
        return df
    
    def plot_errors(self):
        """
        Generates a scatter plot of the predicted and true values, with the prediction errors shown as a separate line plot.
        """
        df = self.generate_error_df()
        p = (ggplot(df, aes(x='y_true', y='y_pred')) +
            geom_point() +
            geom_abline(intercept=0, slope=1, linetype='dashed', color='red') +
            geom_line(aes(x='y_true', y='error'), color='blue') +
            labs(x='True values', y='Predicted values', title='Prediction errors') +
            theme_classic())
        p.draw()
