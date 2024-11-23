import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Load preprocessed data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test) 

# Evaluate Linear Regression
rmse_lin = mean_squared_error(y_test, y_pred_lin, squared=False)
r2_lin = r2_score(y_test, y_pred_lin)
print(f"Linear Regression RMSE: {rmse_lin}")
print(f"Linear Regression R²: {r2_lin}")

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# Evaluate Ridge Regression
rmse_ridge = mean_squared_error(y_test, y_pred_ridge, squared=False)
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f"Ridge Regression RMSE: {rmse_ridge}")
print(f"Ridge Regression R²: {r2_ridge}")

# Set up the figure
plt.figure(figsize=(14, 10))  # Increased size for better visibility

# Linear Regression: Actual vs Predicted
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred_lin, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Linear Regression: Actual vs Predicted Prices')

# Linear Regression: Residuals
plt.subplot(2, 2, 2)
plt.scatter(y_pred_lin, y_test - y_pred_lin, color='blue', alpha=0.5)
plt.axhline(0, color='red', linewidth=2)
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Linear Regression: Residual Plot')

# Ridge Regression: Actual vs Predicted
plt.subplot(2, 2, 3)
plt.scatter(y_test, y_pred_ridge, color='green', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Ridge Regression: Actual vs Predicted Prices')

# Ridge Regression: Residuals
plt.subplot(2, 2, 4)
plt.scatter(y_pred_ridge, y_test - y_pred_ridge, color='green', alpha=0.5)
plt.axhline(0, color='red', linewidth=2)
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Ridge Regression: Residual Plot')

# Ensure layout and save plots if needed
plt.tight_layout()
plt.savefig('regression_analysis_results.png')  # Save the figure for documentation
plt.show()
