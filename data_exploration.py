import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("C:/Users/USER/train.csv")  # Ensure the file path is correct

# Handle missing values
for column in data.select_dtypes(include=["float64", "int64"]).columns:
    data[column].fillna(data[column].median(), inplace=True)
for column in data.select_dtypes(include=["object"]).columns:
    data[column].fillna(data[column].mode()[0], inplace=True)

# ----- Visualizations -----

# Boxplot for detecting outliers in selected columns
selected_columns = ['LotArea', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'SalePrice']
for col in selected_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=data[col])
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)
    plt.show()

# Correlation Heatmap
plt.figure(figsize=(16, 12))  # Increase figure size for better readability
numerical_columns = data.select_dtypes(include=["float64", "int64"])
correlation = numerical_columns.corr()
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", square=False,
            cbar_kws={"shrink": .8}, annot_kws={"size": 8})  # Adjust text size
plt.title("Correlation Heatmap of Numerical Features", fontsize=16)
plt.xticks(rotation=45, fontsize=10)  # Rotate x-axis labels
plt.yticks(fontsize=10)  # Adjust y-axis font size
plt.tight_layout()
plt.show()

# Scatter plots for numerical features vs target
target_column = 'SalePrice'
for col in numerical_columns.columns:
    if col != target_column:  # Exclude the target column
        plt.figure(figsize=(8, 6))
        plt.scatter(data[col], data[target_column], alpha=0.5)
        plt.title(f"Scatter Plot: {col} vs {target_column}")
        plt.xlabel(col)
        plt.ylabel(target_column)
        plt.show()

# Pairplot for relationships between selected numerical features
sns.pairplot(data[selected_columns], diag_kind="kde", corner=True)
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()
