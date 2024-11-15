import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_prepared.load import load
from model.bayesian import bayesian_model

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
    
    return msfe_value

def main():
    training_dataset, testing_dataset = load()
    
    #bayesian model prediction data
    bayesian_predictions=bayesian_model(training_dataset, testing_dataset)
    #bayesian model visualizaiton(model_prediction VS real_Y)
    
    
    
    
    # calculate error
    error_bayesian=msfe(bayesian_predictions, testing_dataset)
    
    # comparing error
    
    
if __name__ == "__main__":
    main()