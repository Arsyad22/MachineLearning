import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

# Load the dataset
data = pd.read_csv("C:/Users/USER/heart.csv")

# ----- Basic Information -----
print("Dataset Info:")
print(data.info())

print("\nDescriptive Statistics:")
print(data.describe())

print("\nFirst few rows of the dataset:")
print(data.head())

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing values in each column:")
print(missing_values[missing_values > 0])

# ----- Visualizations -----

# 1. Correlation Heatmap for Numerical Features
plt.figure(figsize=(10, 8))
correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Heatmap")
plt.show()

# 2. Countplot for Target Variable
target_column = 'condition'  # Replace with the target column name in your dataset
plt.figure(figsize=(6, 4))
sns.countplot(data=data, x=target_column, palette="husl")
plt.title("Distribution of Target Variable")
plt.xlabel("Target")
plt.ylabel("Count")
plt.show()

# 3. Pairplot (Scatter Matrix) for Selected Features
selected_columns = ['age', 'trestbps', 'chol', 'thalach', target_column]  # Modify as needed
sns.pairplot(data[selected_columns], hue=target_column, diag_kind="kde", palette="husl")
plt.suptitle("Scatter Matrix for Age, Chol, Thalach, and Condition", y=1.02)  # Title for the scatter matrix
plt.show()

# 4. Scatter Plots for Individual Features vs Target Variable
numerical_features = ['age', 'trestbps', 'chol', 'thalach']  # Numerical columns
for col in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x=col, y=target_column, hue=target_column, palette="husl")
    plt.title(f"Scatter Plot: {col} vs {target_column}")
    plt.xlabel(col)
    plt.ylabel("Target")
    plt.show()

# 5. Boxplot for Features by Target
for col in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=target_column, y=col, data=data, palette="Set2")
    plt.title(f"Boxplot of {col} by {target_column}")
    plt.xlabel("Target")
    plt.ylabel(col)
    plt.show()

# 6. Scatter Matrix (Alternative using pandas for Selected Features)
scatter_matrix_columns = data[['age', 'chol', 'thalach']]  # Replace as needed
scatter_matrix(scatter_matrix_columns, figsize=(10, 10), diagonal="kde", color="blue")
plt.suptitle("Scatter Matrix for Selected Features")
plt.show()
