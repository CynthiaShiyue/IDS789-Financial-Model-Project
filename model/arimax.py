import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima


def check_stationarity(series: pd.Series) -> bool:
    """
    Check if a time series is stationary using the Augmented Dickey-Fuller test.

    Parameters:
    - series: pd.Series, the time series to test

    Returns:
    - bool: True if the series is stationary, False otherwise
    """
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(series)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] > 0.05:
        print("The series is not stationary.")
        return False
    else:
        print("The series is stationary.")
        return True

def arimax_model(training_dataset: pd.DataFrame, testing_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Train an ARIMAX model and generate predictions.

    Parameters:
    - training_dataset: pd.DataFrame, dataset to train the model
    - testing_dataset: pd.DataFrame, dataset to test the model

    Returns:
    - pd.DataFrame: DataFrame containing Date, Prediction, and Actual values.
    """
    target_var = "UBS log_return"
    exogenous_vars = [
        "Bid-Ask Spread",
        "volume",
        "VIX",
        "EURCHF",
        "oil_log_return",
        "gold_log_return",
        "DB log_return",
        "MS log_return",
        "SPY log_return",
        "^FTSE log_return",
    ]

    # Split data into target and exogenous variables
    y_train = training_dataset[target_var]
    X_train = training_dataset[exogenous_vars]
    y_test = testing_dataset[target_var]
    X_test = testing_dataset[exogenous_vars]

    # Fit ARIMAX model
    print("Finding the best ARIMAX parameters using auto_arima...")
    model = auto_arima(
        y_train,
        X=X_train,
        seasonal=False,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    best_order = model.order
    print(f"Best ARIMAX parameters: {best_order}")

    arimax_model = ARIMA(y_train, order=best_order, exog=X_train, enforce_stationarity=False, enforce_invertibility=False)
    arimax_fit = arimax_model.fit()

    # Forecast on the testing dataset
    y_pred = arimax_fit.forecast(steps=len(y_test), exog=X_test)

     # Prepare output dataset with predictions
    pred_dataset = testing_dataset.copy()
    pred_dataset["Date"] = pred_dataset.index
    pred_dataset["Prediction"] = y_pred.values
    pred_dataset["Actual"] = y_test.values

    return pred_dataset

def plot_predictions_vs_actual(predictions: pd.DataFrame) -> None:
    """
    Plot predictions and actual values with enhanced formatting and save the plot as a PNG file.

    Parameters:
    predictions (pd.DataFrame): DataFrame containing Date, Prediction, and Actual values.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot Actual values
    plt.plot(predictions["Date"], predictions["Actual"], label="Actual", color="blue", linestyle="-", linewidth=1.5)
    
    # Plot Predicted values
    plt.plot(predictions["Date"], predictions["Prediction"], label="Predicted", color="red", linestyle="-", linewidth=1.5)
    
    # Add title and labels
    plt.title("Actual vs Predicted Stock Returns for UBS using ARIMAX", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("UBS Log Return", fontsize=12)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("model/arimax.png")
    print("Plot saved as 'model/arimax.png'")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    # Load pre-split datasets
    training_dataset = pd.read_csv("data_prepared/training_dataset.csv", index_col=0, parse_dates=True)
    testing_dataset = pd.read_csv("data_prepared/testing_dataset.csv", index_col=0, parse_dates=True)

    # Ensure data consistency (e.g., sorting and aligning indices)
    training_dataset.sort_index(inplace=True)
    testing_dataset.sort_index(inplace=True)

    # Fit the ARIMAX model and generate predictions
    predictions = arimax_model(training_dataset, testing_dataset)

    # Plot and save results
    plot_predictions_vs_actual(predictions)
