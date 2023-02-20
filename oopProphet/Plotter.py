from plotnine import ggplot, aes, geom_line, labs
import pandas as pd
from LSTM import LstmTrainer

class Plotter:
    def __init__(self, X_train, y_train, X_test, y_test, scaler, y_pred=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.scaler = scaler
        self.y_pred = y_pred
        self.n_steps = X_train.shape[1]

    def plot_training_data(self):
        train_data = self.scaler.inverse_transform(self.X_train.reshape(-1, self.X_train.shape[-1]))
        train_labels = self.scaler.inverse_transform(self.y_train.reshape(-1, 1))
        train_df = pd.DataFrame({'x': range(self.n_steps, len(train_data)+self.n_steps),
                                 'y': train_data.squeeze(),
                                 'label': ['Training Data'] * len(train_data)})
        train_label_df = pd.DataFrame({'x': range(self.n_steps, len(train_data)+self.n_steps),
                                       'y': train_labels.squeeze(),
                                       'label': ['Training Labels'] * len(train_labels)})
        df = pd.concat([train_df, train_label_df])
        p = (ggplot(df, aes(x='x', y='y', color='label'))
             + geom_line()
             + labs(title='Training Data and Labels', x='Time', y='Value'))
        print(p)

    def plot_testing_data(self):
        test_data = self.scaler.inverse_transform(self.X_test.reshape(-1, self.X_test.shape[-1]))
        test_labels = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        test_df = pd.DataFrame({'x': range(self.n_steps, len(test_data)+self.n_steps),
                                'y': test_data.squeeze(),
                                'label': ['Testing Data'] * len(test_data)})
        test_label_df = pd.DataFrame({'x': range(self.n_steps, len(test_data)+self.n_steps),
                                      'y': test_labels.squeeze(),
                                      'label': ['Testing Labels'] * len(test_labels)})
        df = pd.concat([test_df, test_label_df])
        p = (ggplot(df, aes(x='x', y='y', color='label'))
             + geom_line()
             + labs(title='Testing Data and Labels', x='Time', y='Value'))
        print(p)
    
    def plot_predictions(self):
        test_data = self.scaler.inverse_transform(self.X_test.reshape(-1, self.X_test.shape[-1]))
        test_labels = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        test_df = pd.DataFrame({'x': range(self.n_steps, len(test_data)+self.n_steps),
                                'y': test_data.squeeze(),
                                'label': ['Testing Data'] * len(test_data)})
        test_label_df = pd.DataFrame({'x': range(self.n_steps, len(test_data)+self.n_steps),
                                      'y': test_labels.squeeze(),
                                      'label': ['Testing Labels'] * len(test_labels)})
        pred_df = pd.DataFrame({'x': range(self.n_steps, len(test_data)+self.n_steps),
                                'y': self.y_pred.squeeze(),
                                'label': ['Predictions'] * len(self.y_pred)})
        df = pd.concat([test_df, test_label_df, pred_df])
        p = (ggplot(df, aes(x='x', y='y', color='label'))
             + geom_line()
             + labs(title='Testing Data, Labels, and Predictions', x='Time', y='Value'))
        print(p)

    
    def plot_all(self):
        train_data = self.scaler.inverse_transform(self.X_train.reshape(-1, self.X_train.shape[-1]))
        train_labels = self.scaler.inverse_transform(self.y_train.reshape(-1, 1))
        train_df = pd.DataFrame({'x': range(self.n_steps, len(train_data)+self.n_steps),
                                 'y': train_data.squeeze(),
                                 'label': ['Training Data'] * len(train_data)})
        train_label_df = pd.DataFrame({'x': range(self.n_steps, len(train_data)+self.n_steps),
                                       'y': train_labels.squeeze(),
                                       'label': ['Training Labels'] * len(train_labels)})
        test_data = self.scaler.inverse_transform(self.X_test.reshape(-1, self.X_test.shape[-1]))
        test_labels = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        test_df = pd.DataFrame({'x': range(self.n_steps, len(test_data)+self.n_steps),
                                'y': test_data.squeeze(),
                                'label': ['Testing Data'] * len(test_data)})
        test_label_df = pd.DataFrame({'x': range(self.n_steps, len(test_data)+self.n_steps),
                                      'y': test_labels.squeeze(),
                                      'label': ['Testing Labels'] * len(test_labels)})
        pred_df = pd.DataFrame({'x': range(self.n_steps, len(test_data)+self.n_steps),
                                'y': self.y_pred.squeeze(),
                                'label': ['Predictions'] * len(self.y_pred)})
        df = pd.concat([train_df, train_label_df, test_df, test_label_df, pred_df])
        p = (ggplot(df, aes(x='x', y='y', color='label'))
             + geom_line()
             + labs(title='Training Data, Labels, Testing Data, Labels, and Predictions', x='Time', y='Value'))
        print(p)


if __name__ == '__main__':
    trainer = LstmTrainer('AAPL')
    X_train, y_train, X_test, y_test = trainer.preprocess_data()

    trainer.train()
    y_pred = trainer.generate_predictions()

    # Instantiate the Plotter class with the preprocessed data and predictions
    plotter = Plotter(X_train, y_train, X_test, y_test, trainer.scaler, y_pred)

    # Plot the training data and labels
    plotter.plot_training_data()

    # Plot the testing data, labels, and predictions
    plotter.plot_testing_data()
    plotter.plot_predictions()
