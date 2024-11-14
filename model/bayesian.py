import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_prepared.load import load
import pandas as pd
import numpy as np
import pymc3 as pm  

def bayesian_model(training_dataset, testing_dataset):
    """
    Bayesian model to predict Spread using Volume as a predictor.
    
    Parameters:
    - training_dataset: pd.DataFrame, dataset to train the model
    - testing_dataset: pd.DataFrame, dataset to test the model
    
    Returns:
    - pred_dataset: pd.DataFrame, predictions for testing dataset
    """
    predictions = testing_dataset  # Placeholder for actual predictions
    return predictions

def msfe_bayesian_model(predictions, testing_dataset):
    error = 0  # Placeholder for MSFE calculation
    return error

if __name__ == "__main__":
    # Uncomment the following lines to test the full pipeline
    
    training_dataset, testing_dataset = load()
    # predictions = bayesian_model(training_dataset, testing_dataset)
    # print(predictions)
    
    print(training_dataset.head(5))

