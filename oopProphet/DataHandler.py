import datetime as dt
import os

import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split


'''
A class for handling stock data.

Attributes:
    ticker (str): The name of the stock ticker.
    write_data: A function for writing data to a file.
    test_size (float): The proportion of the data to be used for testing.
    current_day (str): The current date in the format _%Y_%m_%d.
    data (pandas.DataFrame): The data for the given stock ticker,
        either read from a csv file or obtained from the yfinance library
        and saved as a csv file.

Methods:
    __init__(self, ticker, test_size=0.2, write_data=None):
        Initializes a DataHandler instance.
        
        Args:
            ticker (str): The name of the stock ticker.
            test_size (float, optional): The proportion of the data to be used for testing.
            write_data (function, optional): A function for writing data to a file.

    _get_data(self):
        Downloads the data for the given stock ticker using the yfinance library
        and returns it as a pandas DataFrame.
        
        Returns:
            pandas.DataFrame: The stock data.

    _write_data(self, data):
        Writes the given data to a csv file with the name
        ./data/{self.ticker}{self.current_day}.csv.
        
        Args:
            data (pandas.DataFrame): The data to write to a file.

    _split_data(self):
        Splits the data into training and test sets using train_test_split.
        
        Returns:
            Tuple of arrays: X_train, X_test, y_train, y_test.

    _split_data_time_series(self):
        Splits the data into training and test sets based on time.
        
        Returns:
            Tuple of DataFrames: train_data, test_data.

Use:
    data_handler = DataHandler('TSLA')
    data_handler._get_data()
    print(data_handler.data.head())
'''


class DataHandler:
    def __init__(self, ticker, test_size=0.2, write_data=None, model=None):
        self.ticker = ticker
        self.write_data = write_data
        self.test_size = test_size
        self.model = model
        self.current_day = dt.datetime.today().strftime('_%Y_%m_%d')
        try:
            self.data = pd.read_csv(f'./data/{self.ticker}{self.current_day}.csv')
            print(f"Successfully read data for {self.ticker} from csv")
        except Exception as e:
            print(f"Failed to read data for {self.ticker} from csv: {e}")
            self.data = self._get_data()
    
    def _get_data(self):
        data = yf.download(self.ticker, period='2y', interval='1d')
        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
        data = data.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
        return data

    def _write_data(self, data):
        try:
            os.mkdir('./data')
            print("Made data directory")
        except FileExistsError:
            pass
        try:
            df = data
            df.to_csv(f'./data/{self.ticker}{self.current_day}_{self.model}.csv', index=False)
            print(f"Successfully wrote data for {self.ticker} to csv")
        except Exception as e:
            print(f"Failed to write data for {self.ticker} to csv: {e}")

    def split_data(self):
        X = self.data.drop(columns=['y'])
        y = self.data['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
        return X_train, X_test, y_train, y_test

    def split_data_time_series(self):
        n = len(self.data)
        split_idx = int(n * (1 - self.test_size))
        train_data = self.data[:split_idx]
        test_data = self.data[split_idx:]
        return train_data, test_data



if __name__ == '__main__':
    data_handler = DataHandler('TSLA')
    data_handler._get_data()
    print(data_handler.data.head())
    data_handler._write_data(data_handler.data)