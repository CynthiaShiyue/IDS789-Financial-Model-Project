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
    Plot predictions and actual values on the same graph with dates as the x-axis.

    Parameters:
    predictions (pd.DataFrame): Predicted values from the model, including the target variable column.
    testing_dataset (pd.DataFrame): Actual values from the test dataset, including the target variable column.
    """

    target_var = "UBS log_return"

    # Ensure inputs contain the target variable
    if target_var not in testing_dataset.columns:
        raise ValueError(f"Target variable '{target_var}' not found in input datasets.")

    # Ensure the testing dataset index or column contains datetime values
    if "Date" in testing_dataset.columns:
        x = pd.to_datetime(testing_dataset["Date"])
    elif isinstance(testing_dataset.index, pd.DatetimeIndex):
        x = testing_dataset.index
    else:
        raise ValueError(
            "Testing dataset must have a 'Date' column or datetime index for the x-axis."
        )

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(
        x,
        predictions["Prediction"],
        label="Predictions",
        color="blue",
        linestyle="-",
    )
    plt.plot(
        x,
        testing_dataset[target_var],
        label="Actual Y Values",
        color="red",
        linestyle="-",
    )

    # Add titles and labels
    plt.title("Predictions vs Actual Y Values")
    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()


# def plot_results(y_test, y_pred, test_dates, title, output_path=None):
#     """
#     Plots actual vs. predicted values and optionally saves the plot to a file.

#     Parameters:
#     - y_test (pd.Series): Actual target values from the test dataset.
#     - y_pred (np.ndarray): Predicted target values from the model.
#     - test_dates (pd.Series): Date column for x-axis.
#     - title (str): Title for the plot.
#     - output_path (str, optional): Path to save the generated plot. If None, the plot won't be saved.
#     """
#     plt.figure(figsize=(10, 6))

#     # Plot Actual values vs Predicted values
#     plt.plot(test_dates, y_test, label="Actual", color="blue", alpha=0.6)
#     plt.plot(test_dates, y_pred, label="Predicted", color="red", alpha=0.6)

#     # Format x-axis for dates (using DateFormatter for readability)
#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
#     plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
#     plt.xticks(rotation=45)

#     # Add labels and title
#     plt.title(title, fontsize=14)
#     plt.xlabel("Date", fontsize=12)
#     plt.ylabel("Values", fontsize=12)
#     plt.legend()
#     plt.tight_layout()

#     # Save and/or show the plot
#     if output_path:
#         plt.savefig(output_path)
#         print(f"Plot saved to {output_path}")
#     plt.show()
