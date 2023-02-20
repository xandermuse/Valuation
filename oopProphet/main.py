import ModelHandler
import DataHandler
import prophet as Prophet
import time

'''

'''

if __name__ == '__main__':
    # tickerList = ['AAPL', 'AMD', 'VALE', 'AMZN',
    #               'F', 'GM', 'TM', 'TSLA',
    #               'GOOGL', 'META', 'NVDA', 'SPY']

    # create DataHandler object
    ticker = 'TSLA'
    data = DataHandler.DataHandler(ticker)._get_data()

    model = ModelHandler.ModelHandler(data, ticker)
    start = time.time()
    model.run()
    end = time.time()
    print(f"Time taken by the ModelHandler.run() function: {(end - start)/60:.2f} minutes")