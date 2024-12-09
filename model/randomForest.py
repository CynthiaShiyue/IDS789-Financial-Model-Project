import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def train_and_evaluate_rf_model(
    train_path: str, test_path: str, target_column: str
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Train a Random Forest model and evaluate its performance.

    Parameters:
    train_path (str): File path to the training dataset.
    test_path (str): File path to the testing dataset.
    target_column (str): Name of the target variable.

    Returns:
    tuple: A tuple containing the test dataset dates, actual target values, and predicted values.
    """
    # Load datasets
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Ensure Date is in datetime format
    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])

    # Prepare training and testing data
    X_train = train.drop(columns=[target_column, "Date"])
    y_train = train[target_column]
    X_test = test.drop(columns=[target_column, "Date"])
    y_test = test[target_column]

    # Initialize Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

    # Train the model
    rf_model.fit(X_train, y_train)

    # Predict on test data
    y_pred = rf_model.predict(X_test)

    # Evaluate model performance
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Random Forest MAE: {mae:.4f}")
    print(f"Random Forest MSE: {mse:.4f}")
    print(f"Random Forest R² Score: {r2:.4f}")

    return test["Date"], y_test, y_pred


def plot_predictions(
    test_dates: pd.Series, y_test: pd.Series, y_pred: pd.Series, output_path: str
) -> None:
    """
    Plot actual vs predicted values for the test dataset.

    Parameters:
    test_dates (pd.Series): Dates corresponding to the test data.
    y_test (pd.Series): Actual target values.
    y_pred (pd.Series): Predicted target values.
    output_path (str): File path to save the plot.
    """
    plt.figure(figsize=(10, 6))

    # Plot Actual and Predicted values
    plt.plot(test_dates, y_test, label="Actual", color="blue", alpha=0.6)
    plt.plot(test_dates, y_pred, label="Predicted", color="red", alpha=0.6)

    # Format x-axis for dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    # Add labels and title
    plt.title(
        "Actual vs Predicted Stock Returns for UBS using Random Forest", fontsize=14
    )
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("UBS Log Return", fontsize=12)
    plt.legend()
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(output_path)
    plt.show()


train_path = r"C:\Users\DELL\OneDrive\Desktop\IDS789-Financial-Model-Project\data_prepared\training_dataset.csv"
test_path = r"C:\Users\DELL\OneDrive\Desktop\IDS789-Financial-Model-Project\data_prepared\testing_dataset.csv"
output_path = r"C:\Users\DELL\OneDrive\Desktop\IDS789-Financial-Model-Project\model\randomforest.png"

# Train model and get predictions
test_dates, y_test, y_pred = train_and_evaluate_rf_model(
    train_path=train_path, test_path=test_path, target_column="UBS log_return"
)

# Plot results
plot_predictions(
    test_dates=test_dates, y_test=y_test, y_pred=y_pred, output_path=output_path
)


# rf = RandomForestRegressor()

# # Set up hyperparameters to tune
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 20, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'bootstrap': [True, False]
# }

# # GridSearchCV to find the best hyperparameters
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# # Fit the model
# grid_search.fit(X_train, y_train)

# # Print the best hyperparameters
# print(f"Best Hyperparameters: {grid_search.best_params_}")

# # Use the best estimator to predict
# best_rf = grid_search.best_estimator_
# y_pred = best_rf.predict(X_test)
