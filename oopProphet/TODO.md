Models to be added

- Autoregressive Integrated Moving Average (ARIMA): ARIMA is a popular time series forecasting model that can be used to model the trends and seasonality in the data. ARIMA is a widely used model in financial forecasting due to its ability to model non-stationary time series.

- Long Short-Term Memory (LSTM): LSTM is a type of recurrent neural network that can be used for time series forecasting. LSTMs are particularly good at capturing long-term dependencies in the data, and have been used successfully in stock price forecasting.

- Random Forest (RF): Random Forest is a popular machine learning algorithm that can be used for regression problems, including stock price forecasting. Random Forest models can handle both linear and nonlinear relationships in the data, and can be particularly useful when dealing with high-dimensional data.

- Gradient Boosting (GBM): Gradient Boosting is another machine learning algorithm that has been used successfully for stock price forecasting. GBMs are particularly good at handling noisy data and can often outperform other machine learning algorithms in terms of accuracy.

- Gaussian Process Regression (GPR): GPR is a probabilistic machine learning model that can be used for regression problems. GPR is particularly good at handling small data sets, and can be used to model both linear and nonlinear relationships in the data.

- It's important to note that no single model can always produce accurate forecasts. It is usually recommended to compare the performance of multiple models to determine the best fit for the specific forecasting task. Additionally, it's important to incorporate domain knowledge and market understanding to generate better forecasts.




Hyperparameter Tuning methods.

- The choice of parameter tuning technique depends on the specific model being used for stock price forecasting, as well as the available data and computational resources. Here are some commonly used parameter tuning techniques:

    - Grid Search: Grid search is a simple and widely used method for hyperparameter tuning. It involves specifying a grid of hyperparameter values, and exhaustively searching over the grid to find the best hyperparameter combination that yields the best performance on a validation set.

    - Random Search: Random search is an alternative to grid search that can be more efficient when dealing with a large number of hyperparameters. Random search involves randomly sampling hyperparameters from a distribution, and evaluating their performance on a validation set.

    - Bayesian Optimization: Bayesian optimization is a more advanced method for hyperparameter tuning that uses probabilistic models to guide the search for the best hyperparameters. Bayesian optimization can be more efficient than grid search and random search when the number of hyperparameters is large, or when the evaluation of the model is time-consuming.

    - Genetic Algorithms: Genetic algorithms are another optimization technique that can be used for hyperparameter tuning. Genetic algorithms involve encoding the hyperparameters as genes, and iteratively generating new generations of hyperparameter combinations through mutation and crossover. Genetic algorithms can be useful when dealing with complex optimization problems, and can often find good solutions even in high-dimensional search spaces.

- The choice of parameter tuning technique depends on the specific requirements of the forecasting task and the available resources. Grid search and random search are simple and effective methods that are easy to implement, while Bayesian optimization and genetic algorithms can be more powerful but require more computational resources. It's recommended to try multiple hyperparameter tuning techniques and evaluate their performance to choose the best one for the specific forecasting task.

|Hyperparameter Tuning methods| status |
|:-|:-|
|Grid Search|Done|
|Random Search|Not Started|
|Bayesian Optimization|Not Started|
|Genetic Algorithms|Not Started|

|Models| status |Takes|Output|
|:-|:-|:-|:-|
|Autoregressive Integrated Moving Average (ARIMA)|Not Started|
|Long Short-Term Memory (LSTM)|Not Started|
|Random Forest (RF)|Not Started|
|Gradient Boosting (GBM)|Not Started|
|Gaussian Process Regression (GPR)|Not Started|
|Prophet|Done|