import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.dates as mdates


def msfe(predictions: pd.DataFrame, testing_dataset: pd.DataFrame) -> float:
    """
    Calculate the Mean Squared Forecast Error (MSFE).

    Parameters:
    predictions (pd.DataFrame): Predicted values from the model, including the target variable column.
    testing_dataset (pd.DataFrame): Actual values from the test dataset, including the target variable column.

    Returns:
    float: The Mean Squared Forecast Error.
    """
    target_var = "UBS log_return"

    # Ensure inputs contain the target variable
    if (
        target_var not in predictions.columns
        or target_var not in testing_dataset.columns
    ):
        raise ValueError(f"Target variable '{target_var}' not found in input datasets.")

    # Calculate squared errors
    squared_errors = (predictions["Prediction"] - testing_dataset[target_var]) ** 2

    # Calculate the mean of squared errors
    msfe_value = squared_errors.mean()

    print(f"Mean Squared Forecast Error: {msfe_value:.6f}")
    return msfe_value


def plot_predictions_vs_actual(
    predictions: pd.DataFrame, testing_dataset: pd.DataFrame
) -> None:
    """
    Plot predictions and actual values on the same graph with different colors.

    Parameters:
    predictions (pd.DataFrame): DataFrame containing predicted values, including a 'Date' column and 'Prediction' column.
    testing_dataset (pd.DataFrame): DataFrame containing actual values, including a 'Date' column and target variable column.
    """
    target_var = "UBS log_return"

    # Ensure inputs contain the required columns
    if "Date" not in predictions.columns or "Date" not in testing_dataset.columns:
        raise ValueError("Both input datasets must contain a 'Date' column.")
    if target_var not in testing_dataset.columns:
        raise ValueError(
            f"Target variable '{target_var}' not found in the testing dataset."
        )
    if "Prediction" not in predictions.columns:
        raise ValueError("'Predictions' DataFrame must contain a 'Prediction' column.")

    # Ensure Date columns are in datetime format
    predictions["Date"] = pd.to_datetime(predictions["Date"])
    testing_dataset["Date"] = pd.to_datetime(testing_dataset["Date"])

    # Merge the dataframes on the Date column for consistent x-axis
    merged_data = pd.merge(predictions, testing_dataset, on="Date")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        merged_data["Date"],
        merged_data["Prediction"],
        label="Predicted",
        color="blue",
        linestyle="--",
        alpha=0.7,
    )
    plt.plot(
        merged_data["Date"],
        merged_data[target_var],
        label="Actual",
        color="red",
        linestyle="-",
        alpha=0.7,
    )

    # Format x-axis for dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    # Add labels and title
    plt.title("Actual vs Predicted Stock Returns for UBS", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("UBS Log Return", fontsize=12)
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()
