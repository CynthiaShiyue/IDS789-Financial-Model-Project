import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# from sklearn.model_selection import GridSearchCV

# Load training and test datasets
train = pd.read_csv(
    r"C:\Users\DELL\OneDrive\Desktop\IDS789-Financial-Model-Project\data_prepared\training_dataset.csv"
)
test = pd.read_csv(
    r"C:\Users\DELL\OneDrive\Desktop\IDS789-Financial-Model-Project\data_prepared\testing_dataset.csv"
)

# Ensure Date is in datetime format
train["Date"] = pd.to_datetime(train["Date"])
test["Date"] = pd.to_datetime(test["Date"])

# target-explanatory variables split
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

plt.figure(figsize=(10, 6))

# Plot Actual values vs Predicted values
plt.plot(test["Date"], y_test, label="Actual", color="blue", alpha=0.6)
plt.plot(test["Date"], y_pred, label="Predicted", color="red", alpha=0.6)

# Format x-axis for dates (using DateFormatter for readability)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)

# Add labels and title
plt.title("Actual vs Predicted Stock Returns for UBS using Decision Trees", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("UBS Log Return", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig(
    r"C:\Users\DELL\OneDrive\Desktop\IDS789-Financial-Model-Project\model\decisiontree.png"
)

# Show the plot
plt.show()

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
