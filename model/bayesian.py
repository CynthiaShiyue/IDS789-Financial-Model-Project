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
    Bayesian model to predict UBS's return using predictors and weekday-based hierarchical prior.

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
    
    # Add weekday column to datasets
    training_dataset["weekday"] = training_dataset["Date"].dt.weekday  # Monday=0, Sunday=6
    testing_dataset["weekday"] = testing_dataset["Date"].dt.weekday

    # Extract training data
    X_train = training_dataset[predictors].values
    y_train = training_dataset[target_var].values
    weekdays_train = training_dataset["weekday"].values

    # Standardize predictors for stability
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    X_train_normalized = (X_train - X_train_mean) / X_train_std

    # Standardize test data using training data parameters
    X_test = testing_dataset[predictors].values
    weekdays_test = testing_dataset["weekday"].values
    X_test_normalized = (X_test - X_train_mean) / X_train_std

    # Define Bayesian model with hierarchical priors for weekday effects
    with pm.Model() as model:
        # Hyperpriors for weekday-specific intercepts
        weekday_mu = pm.Normal("weekday_mu", mu=0, sigma=10)  # Shared mean for weekdays
        weekday_sigma = pm.HalfNormal("weekday_sigma", sigma=10)  # Variability across weekdays

        # Weekday-specific intercepts
        alpha_weekday = pm.Normal(
            "alpha_weekday", mu=weekday_mu, sigma=weekday_sigma, shape=7
        )  # One intercept per weekday (0-6)

        # Coefficients for predictors
        beta = pm.Normal("beta", mu=0, sigma=10, shape=len(predictors))
        sigma = pm.HalfNormal("sigma", sigma=10)

        # Linear model
        weekday_effect = alpha_weekday[weekdays_train]
        mu = weekday_effect + pm.math.dot(X_train_normalized, beta)

        # Likelihood
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)

        # Sample from the posterior
        trace = pm.sample(2000, tune=1000, cores=2)

        # Extract posterior means
        pred_alpha_weekday = np.mean(trace.posterior["alpha_weekday"].values, axis=(0, 1))
        pred_beta = np.mean(trace.posterior["beta"].values, axis=(0, 1))

    # Generate predictions for the test set
    weekday_effect_test = pred_alpha_weekday[weekdays_test]
    y_pred = weekday_effect_test + np.dot(X_test_normalized, pred_beta)

    # Combine predictions with the testing dataset
    pred_dataset = testing_dataset.copy()
    pred_dataset["Prediction"] = y_pred

    return pred_dataset



if __name__ == "__main__":
    training_dataset, testing_dataset = load()
    predictions_df = bayesian_model(training_dataset, testing_dataset)
    #print(predictions_df.head(5))
    #plot_predictions_vs_actual(predictions_df, testing_dataset)
    print(msfe(predictions_df, testing_dataset))
    
