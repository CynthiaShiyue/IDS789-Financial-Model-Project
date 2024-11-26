import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


def msfe(predictions: pd.DataFrame, testing_dataset: pd.DataFrame) -> float:
    """
    Calculate the Mean Squared Forecast Error (MSFE).

    Parameters:
    predictions (pd.DataFrame): Predicted values from the model, including the target variable column.
    testing_dataset (pd.DataFrame): Actual values from the test dataset, including the target variable column.

    Returns:
    float: The Mean Squared Forecast Error.
    """
    target_var = 'UBS log_return'
    
    # Ensure inputs contain the target variable
    if target_var not in predictions.columns or target_var not in testing_dataset.columns:
        raise ValueError(f"Target variable '{target_var}' not found in input datasets.")
    
    # Calculate squared errors
    squared_errors = (predictions["Prediction"] - testing_dataset[target_var]) ** 2
    
    # Calculate the mean of squared errors
    msfe_value = squared_errors.mean()
    
    print(f"Mean Squared Forecast Error: {msfe_value:.6f}")
    return msfe_value


def plot_predictions_vs_actual(predictions: pd.DataFrame, testing_dataset: pd.DataFrame) -> None:
    """
    Plot predictions and actual values on the same graph with different colors.

    Parameters:
    predictions (pd.DataFrame): Predicted values from the model, including the target variable column.
    testing_dataset (pd.DataFrame): Actual values from the test dataset, including the target variable column.
    """
    target_var = 'UBS log_return'
    
    # Ensure inputs contain the target variable
    if target_var not in predictions.columns or target_var not in testing_dataset.columns:
        raise ValueError(f"Target variable '{target_var}' not found in input datasets.")
    
    # Create an index for the x-axis
    x = range(len(predictions))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, predictions["Prediction"], label='Predictions', color='blue', marker='o', linestyle='--')
    plt.plot(x, testing_dataset[target_var], label='Actual Y Values', color='red', marker='o', linestyle='-')
    
    # Add titles and labels
    plt.title('Predictions vs Actual Y Values')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()
