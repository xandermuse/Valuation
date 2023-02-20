import DataHandler
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV
from plotnine import *
import json
import multiprocessing
import scipy.stats as stats


class LstmTrainer:
    def __init__(self, ticker):
        self.data = DataHandler.DataHandler(ticker)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.y_pred = None
        self.y_test = None
        self.n_features = None
        self.n_steps = 7 
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.ticker = ticker
        print("LSTM Trainer initialized")

    def preprocess_data(self):
        train_data, test_data = self.data.split_data_time_series()
        train_data = train_data[['ds', 'y']]
        test_data = test_data[['ds', 'y']]
        train_data_scaled = self.scaler.fit_transform(train_data[['y']])
        self.n_features = train_data_scaled.shape[1]
        X_train, y_train = self.reshape_data(train_data_scaled)
        X_test, y_test = self.reshape_data(self.scaler.transform(test_data[['y']]))
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.train_dates = train_data[self.n_steps:]['ds']
        self.test_dates = test_data[self.n_steps:]['ds']
        return X_train, y_train, X_test, y_test


    def reshape_data(self, data):
        X = []
        y = []
        for i in range(self.n_steps, len(data)):
            X.append(data[i - self.n_steps:i, :])
            y.append(data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], self.n_steps, self.n_features))
        return X, y

    def create_model(self, n_neurons=100, n_epochs=100, batch_size=32, dropout_rate=0.2, l2_regularization=0.001):
        optimizer = Adam(learning_rate=0.001)
        self.model = Sequential()
        self.model.add(LSTM(units=n_neurons, input_shape=(self.n_steps, self.n_features), return_sequences=True))
        self.model.add(Dropout(dropout_rate))
        self.model.add(LSTM(units=n_neurons, return_sequences=True))
        self.model.add(Dropout(dropout_rate))
        self.model.add(LSTM(units=n_neurons, return_sequences=False))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(units=25, activation='linear', kernel_regularizer=l2(l2_regularization)))
        self.model.add(Dense(units=1, activation='linear', kernel_regularizer=l2(l2_regularization)))
        self.model.compile(loss='mse', optimizer=optimizer)
        self.model.fit(self.X_train, self.y_train, epochs=n_epochs, batch_size=batch_size, verbose=1)


    def run_experiments(self, param_grid):
        with multiprocessing.Pool() as pool:
            results = [pool.apply_async(self.train, kwds=exp) for exp in ParameterGrid(param_grid)]
            pool.close()
            pool.join()

        return [result.get() for result in results]

    def random_search(self, param_dist, n_iter):
        self.preprocess_data()

        lstm_model = KerasRegressor(build_fn=self.create_model, verbose=0)
        search = RandomizedSearchCV(lstm_model, param_distributions=param_dist, n_iter=n_iter, cv=5, verbose=1)
        search_result = search.fit(self.X_train, self.y_train)

        self.model = search_result.best_estimator_.model
        self.predict()
        mse = self.evaluate()

        return search_result.best_params_, mse

    def grid_search(self, param_grid):
        results = self.run_experiments(param_grid)
        sorted_results = sorted(results, key=lambda x: x[1]) # sort by MSE

        with open(f'{self.ticker}_results.json', 'w') as f:
            for result in sorted_results:
                json.dump(result, f)
                f.write('\n')


    def use_best_params(self):
        with open(f'{self.ticker}_results.json', 'r') as f:
            best_params, _ = json.loads(f.readline())
        self.train(*best_params)


    def evaluate(self):
        y_test_subset = self.y_test[self.n_steps:self.n_steps+len(self.y_pred)]
        mse = mean_squared_error(y_test_subset, self.y_pred[:, 0])
        print(f"Test MSE: {mse}")
        return mse
    
    def predict(self):
        X_test = self.X_test.reshape((self.X_test.shape[0], self.n_steps, self.n_features))
        y_pred_all = self.model.predict(X_test)
        y_pred_all = y_pred_all.reshape(y_pred_all.shape[0], -1)
        self.y_pred = self.scaler.inverse_transform(y_pred_all)[:len(self.y_test) - self.n_steps]

    def train(self, n_neurons=100, n_epochs=100, batch_size=32, dropout_rate=0.2, l2_regularization=0.001):
        self.preprocess_data()
        self.create_model(n_neurons=n_neurons, n_epochs=n_epochs, batch_size=batch_size, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.predict()
        mse = self.evaluate()
        return (n_neurons, n_epochs, batch_size, dropout_rate, l2_regularization), mse


    def generate_predictions(self):
        X_test = self.X_test.reshape((self.X_test.shape[0], self.n_steps, self.n_features))
        y_pred_all = self.model.predict(X_test)
        y_pred_all = y_pred_all.reshape(y_pred_all.shape[0], -1)
        self.y_pred = self.scaler.inverse_transform(y_pred_all)[:len(self.y_test) - self.n_steps]

    def plot(self, show_plot=False, save_file=None):
        predictions = pd.DataFrame(self.y_pred, columns=['y_pred'])
        predictions['y_true'] = self.y_test[self.n_steps:self.n_steps+len(self.y_pred)]
        predictions = predictions.reset_index().melt(id_vars='index', var_name='variable', value_name='value')

        p = (ggplot(predictions, aes(x='index', y='value', color='variable')) +
            geom_line() +
            labs(x='Time', y='Value', color='') +
            theme_classic())

        if save_file is not None:
            ggsave(filename=save_file, plot=p, dpi=300)

        if show_plot:
            p.draw()

        return p




if __name__ == '__main__':
    lstm_trainer = LstmTrainer('TSLA')
    # param_grid = {
    #     'n_neurons': [50, 100, 200],
    #     'n_epochs': [50, 100, 200],                 
    #     'batch_size': [16, 32, 64],
    #     'dropout_rate': [0.1, 0.2, 0.3],
    #     'l2_regularization': [0.0001, 0.001, 0.01],
    # }
    import tensorflow as tf
    print(tf.__version__)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print([gpu.name for gpu in gpus])
    
    param_grid = {
    'n_neurons': [200, 300, 400],
    'n_epochs': [100, 200, 300],
    'batch_size': [16, 32, 64],
    'dropout_rate': [0.1, 0.2, 0.3],
    'l2_regularization': [0.00001, 0.0001, 0.001],
    }


    param_dist = {
        'n_neurons': stats.randint(5, 20),
        'n_epochs': stats.randint(5, 20),
        'batch_size': [16],
        'dropout_rate': stats.uniform(0.1, 0.3),
        'l2_regularization': stats.loguniform(1e-6, 1e-3),
    }
    # lstm_trainer.grid_search(param_grid)
    lstm_trainer.random_search(param_dist, n_iter=1)
    
    # Error analysis
    

# There are several ways to potentially improve the LstmTrainer class:

# Hyperparameter tuning: The create_model method currently takes in a 
# few hyperparameters such as the number of neurons in the LSTM layers, 
# number of epochs, and batch size. These hyperparameters can be fine-tuned 
# to improve the performance of the model. One way to do this is to use 
# grid search or random search to find the optimal hyperparameters.

# Regularization: The current LSTM model architecture does not include 
# any regularization techniques such as dropout or L2 regularization. 
# These techniques can be added to the model to reduce overfitting and 
# improve its generalization performance.

# Ensembling: One way to improve the performance of the model is to train 
# multiple LSTM models with different hyperparameters or initializations 
# and then ensemble their predictions. This can lead to better prediction 
# accuracy and more robustness.

# Feature engineering: The LstmTrainer class only uses the stock's closing 
# price as input. It may be beneficial to incorporate other features such 
# as volume, technical indicators, or other financial indicators to improve 
# the model's performance.

# Error analysis: The evaluate method currently only reports the mean squared error 
# between the predicted and actual values of the test data. It may be useful to 
# perform a more detailed error analysis to understand which data points the model 
# is struggling to predict and why. This can lead to insights on how to further 
# improve the model.