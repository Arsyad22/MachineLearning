import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("C:/Users/USER/train.csv")  

# Drop columns with too many missing values
data = data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

# Fill missing values for numerical columns with mean
data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].mean())
data['MasVnrArea'] = data['MasVnrArea'].fillna(data['MasVnrArea'].mean())

# Fill missing values for categorical columns with mode
data['BsmtQual'] = data['BsmtQual'].fillna(data['BsmtQual'].mode()[0])
data['GarageType'] = data['GarageType'].fillna(data['GarageType'].mode()[0])

# Fill missing values for 'GarageYrBlt' with the median
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['GarageYrBlt'].median())

# Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(data, drop_first=True)

# Split the data into features and target variable
X = data.drop('SalePrice', axis=1)  # Features
y = data['SalePrice']  # Target

# Check for any remaining missing values
print("Missing values in X:")
print(X.isnull().sum()[X.isnull().sum() > 0])
print("Missing values in y:")
print(y.isnull().sum())

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the training and testing sets as CSV files
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Preprocessing complete! Training and testing data saved as CSV files.")
