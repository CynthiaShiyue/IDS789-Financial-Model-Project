import matplotlib.pyplot as plt

def msfe(predictions, testing_dataset):
    """
    Calculate the Mean Squared Forecast Error (MSFE).
    
    Parameters:
    predictions: Predicted values from the model.
    testing_dataset: Actual values from the test dataset.
    
    Returns:
    float: The Mean Squared Forecast Error.
    """
    # Ensure inputs are numpy arrays for element-wise operations
    predictions = predictions
    testing_dataset = testing_dataset
    
    # Calculate squared errors
    squared_errors = (predictions - testing_dataset) ** 2
    
    # Calculate the mean of squared errors
    msfe_value = squared_errors.mean()
    
    print(f"Mean Squared Forecast Error: {msfe_value}")
    return msfe_value



def plot_predictions_vs_actual(predictions, testing_dataset):
    """
    Plot predictions and actual values on the same graph with different colors.

    Parameters:
    predictions (list or array-like): Predicted values from the model.
    testing_dataset (list or array-like): Actual values from the test dataset.
    """
    # Create an index for the x-axis
    x = range(len(predictions))

    # Plot predictions
    plt.plot(x, predictions, label='Predictions', color='blue', marker='o', linestyle='--')

    # Plot actual values
    plt.plot(x, testing_dataset, label='Actual Y Values', color='red', marker='x', linestyle='-')

    # Add titles and labels
    plt.title('Predictions vs Actual Y Values')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()



