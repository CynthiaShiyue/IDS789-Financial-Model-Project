import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# from sklearn.model_selection import GridSearchCV


def fit_decision_tree(train_path, test_path):
    """
    Fits a Decision Tree Regressor on the training data and evaluates it on the test data.

    Parameters:
    - train_path (str): Path to the training dataset.
    - test_path (str): Path to the testing dataset.

    Returns:
    - y_test (pd.Series): Actual target values from the test dataset.
    - y_pred (np.ndarray): Predicted target values from the Decision Tree model.
    - test_dates (pd.Series): Date column from the test dataset for plotting.
    """
    # Load training and test datasets
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Ensure Date is in datetime format
    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])

    # Target-explanatory variables split
    X_train = train.drop(columns=["UBS log_return", "Date"])
    y_train = train["UBS log_return"]
    X_test = test.drop(columns=["UBS log_return", "Date"])
    y_test = test["UBS log_return"]

    # Train the Decision Tree model
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print Results
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Forecast Error (MSFE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return y_test, y_pred, test["Date"]


def plot_results(y_test, y_pred, test_dates, output_path):
    """
    Plots actual vs. predicted values and saves the plot to a file.

    Parameters:
    - y_test (pd.Series): Actual target values from the test dataset.
    - y_pred (np.ndarray): Predicted target values from the model.
    - test_dates (pd.Series): Date column for x-axis.
    - output_path (str): Path to save the generated plot.
    """
    plt.figure(figsize=(10, 6))

    # Plot Actual values vs Predicted values
    plt.plot(test_dates, y_test, label="Actual", color="blue", alpha=0.6)
    plt.plot(test_dates, y_pred, label="Predicted", color="red", alpha=0.6)

    # Format x-axis for dates (using DateFormatter for readability)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    # Add labels and title
    plt.title(
        "Actual vs Predicted Stock Returns for UBS using Decision Trees", fontsize=14
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
output_path = r"C:\Users\DELL\OneDrive\Desktop\IDS789-Financial-Model-Project\model\decisiontree.png"

# Fit model and get results
y_test, y_pred, test_dates = fit_decision_tree(train_path, test_path)

# Plot results
plot_results(y_test, y_pred, test_dates, output_path)


# # Set up hyperparameter grid
# param_grid = {
#     "max_depth": [3, 5, 10, None],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
# }

# # Initialize model and GridSearchCV
# model = DecisionTreeRegressor(random_state=42)
# grid_search = GridSearchCV(
#     estimator=model, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error"
# )

# # Fit grid search
# grid_search.fit(X_train, y_train)

# # Get the best model
# best_model = grid_search.best_estimator_
# print("Best Parameters:", grid_search.best_params_)

# # Predict and evaluate
# y_pred = best_model.predict(X_test)
# print("MAE:", mean_absolute_error(y_test, y_pred))
# print("MSE:", mean_squared_error(y_test, y_pred))
