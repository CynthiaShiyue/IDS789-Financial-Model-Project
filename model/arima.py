import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def check_stationarity(series):
    """
    Check if a time series is stationary using the Augmented Dickey-Fuller test.

    Parameters:
    - series: pd.Series, the time series to test

    Returns:
    - bool: True if the series is stationary, False otherwise
    """
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] > 0.05:
        print("The series is not stationary.")
        return False
    else:
        print("The series is stationary.")
        return True


def arima_model(training_dataset, testing_dataset):
    """
    ARIMA model to predict future values using historical log returns.
    Automatically determine the best parameters using auto_arima.

    Parameters:
    - training_dataset: pd.Series, dataset to train the model
    - testing_dataset: pd.Series, dataset to test the model

    Returns:
    - forecast: pd.Series, predictions for testing dataset
    """
    # Use auto_arima to determine the best parameters
    print("Finding the best ARIMA parameters using auto_arima...")
    model = auto_arima(training_dataset, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    best_order = model.order
    print(f"Best ARIMA parameters: {best_order}")

    # Fit the ARIMA model with the best parameters
    model = ARIMA(training_dataset, order=best_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()

    # Forecasting the next len(testing_dataset) periods
    forecast = model_fit.forecast(steps=len(testing_dataset))

    # Evaluate the model using testing dataset
    mse = mean_squared_error(testing_dataset, forecast)
    mae = mean_absolute_error(testing_dataset, forecast)
    rmse = np.sqrt(mse)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Residual Analysis
    residuals = model_fit.resid
    plt.figure(figsize=(10, 5))
    plt.plot(residuals)
    plt.title("Residuals of the ARIMA Model")
    plt.show()
    plot_acf(residuals, lags=30)
    plt.title("ACF of Residuals")
    plt.show()
    plot_pacf(residuals, lags=30)
    plt.title("PACF of Residuals")
    plt.show()

    # Plotting actual vs predicted values
    plt.figure(figsize=(10, 5))
    plt.plot(testing_dataset.index, testing_dataset, label='Actual', color='blue')
    plt.plot(testing_dataset.index, forecast, label='Predicted', color='red')
    plt.title('Actual vs Predicted Log Returns')
    plt.xlabel('Date')
    plt.ylabel('Log Return')
    plt.legend()
    plt.show()

    return forecast


if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/UBS.csv', parse_dates=True, index_col=0)

    # Ensure frequency for date index
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('B', method='pad')  # Assuming business day frequency and padding missing values if needed

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['adjclose'], label='Adjusted Close Price', color='blue')
    plt.title('Original Adjusted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Plot the original time series along with a rolling mean
    rolling_window = 30  # Set a suitable window size, e.g., 30 days

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['adjclose'], label='Adjusted Close Price', color='blue')
    plt.plot(df.index, df['adjclose'].rolling(window=rolling_window).mean(), label='Rolling Mean (30 days)', color='red')
    plt.title('Adjusted Close Price with Rolling Mean')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Calculate log returns
    log_return = np.log(df['adjclose'] / df['adjclose'].shift(1))
    log_return = log_return.dropna()

    # Check for stationarity
    if not check_stationarity(log_return):
        log_return = log_return.diff().dropna()  # Differencing to make the series stationary
        check_stationarity(log_return)  # Recheck stationarity

    # Split the data into training and testing datasets
    split_point = int(len(log_return) * 0.8)
    training_dataset = log_return[:split_point]
    testing_dataset = log_return[split_point:]

    # Fit the ARIMA model and predict
    try:
        predictions = arima_model(training_dataset, testing_dataset)
        print(predictions)
    except Exception as e:
        print(f"Error: {e}")
        