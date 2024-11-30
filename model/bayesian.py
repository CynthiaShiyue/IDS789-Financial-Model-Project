import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import pymc as pm
from model.lib import plot_predictions_vs_actual, msfe

# Ensure the 'load' function can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_prepared.load import load


def bayesian_model(training_dataset, testing_dataset):
    """
    Bayesian model to predict UBS's return using predictors

    Parameters:
    - training_dataset: pd.DataFrame, dataset to train the model
    - testing_dataset: pd.DataFrame, dataset to test the model

    Returns:
    - pred_dataset: pd.DataFrame, predictions for testing dataset
    """
    predictors = [
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
    target_var = "UBS log_return"

    # Extract training data
    X_train = training_dataset[predictors].values
    y_train = training_dataset[target_var].values

    # Standardize predictors for stability
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    X_train_normalized = (X_train - X_train_mean) / X_train_std

    # Standardize test data using training data parameters
    X_test = testing_dataset[predictors].values
    X_test_normalized = (X_test - X_train_mean) / X_train_std

    # Define Bayesian model
    with pm.Model() as model:
        # Priors for intercept and coefficients
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=len(predictors))
        sigma = pm.HalfNormal("sigma", sigma=10)

        # Linear model
        mu = alpha + pm.math.dot(X_train_normalized, beta)

        # Likelihood
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)

        # Sample from the posterior
        trace = pm.sample(2000, tune=1000, cores=2)

        # Extract posterior means of alpha and beta
        pred_alpha = np.mean(trace.posterior["alpha"].values)
        pred_beta = np.mean(trace.posterior["beta"].values, axis=(0, 1))

    # Generate predictions for the test set
    y_pred = pred_alpha + np.dot(X_test_normalized, pred_beta)
    

    # Combine predictions with the testing dataset
    pred_dataset = testing_dataset.copy()
    pred_dataset["Prediction"] = y_pred

    return pred_dataset


if __name__ == "__main__":
    training_dataset, testing_dataset = load()
    predictions_df = bayesian_model(training_dataset, testing_dataset)

    plot_predictions_vs_actual(predictions_df, testing_dataset)
    print(msfe(predictions_df, testing_dataset))
    
