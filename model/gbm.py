import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from model.lib import plot_predictions_vs_actual, msfe

def gbm_model(training_dataset, testing_dataset):
    """
    Gradient Boosting Machine model to predict UBS's return using predictors

    Parameters:
    - training_dataset: pd.DataFrame, dataset to train the model
    - testing_dataset: pd.DataFrame, dataset to test the model

    Returns:
    - pred_dataset: pd.DataFrame, predictions for testing dataset
    """
    # Define predictors (same as in Bayesian model for consistency)
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

    # Prepare training data
    X_train = training_dataset[predictors]
    y_train = training_dataset[target_var]
    
    # Prepare testing data
    X_test = testing_dataset[predictors]
    y_test = testing_dataset[target_var]

    # Initialize and train GBM model
    gbm = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42
    )
    
    # Fit the model
    gbm.fit(X_train, y_train)
    
    # Make predictions
    y_pred = gbm.predict(X_test)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"R-squared Score: {r2:.6f}")

    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': predictors,
        'importance': gbm.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['feature'], feature_importance['importance'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Importance in GBM Model')
    plt.tight_layout()
    plt.show()

    # Learning curves analysis
    train_scores = []
    test_scores = []
    n_estimators = np.arange(10, 110, 10)
    
    for n in n_estimators:
        model = GradientBoostingRegressor(
            n_estimators=n,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        model.fit(X_train, y_train)
        train_scores.append(mean_squared_error(y_train, model.predict(X_train)))
        test_scores.append(mean_squared_error(y_test, model.predict(X_test)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators, train_scores, label='Training MSE')
    plt.plot(n_estimators, test_scores, label='Testing MSE')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curves')
    plt.legend()
    plt.show()

    # Prepare output dataset with predictions
    pred_dataset = testing_dataset.copy()
    pred_dataset["Prediction"] = y_pred

    return pred_dataset

if __name__ == "__main__":
    # Load data (assuming similar structure to other models)
    training_dataset = pd.read_csv("data_prepared/training_dataset.csv")
    testing_dataset = pd.read_csv("data_prepared/testing_dataset.csv")
    
    # Run GBM model
    predictions_df = gbm_model(training_dataset, testing_dataset)
    
    # Plot predictions vs actual
    plot_predictions_vs_actual(predictions_df, testing_dataset)
    
    # Calculate MSFE
    error = msfe(predictions_df, testing_dataset)
    print(f"\nMean Squared Forecast Error: {error:.6f}")
