import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('C:/Users/USER/heart.csv')

# Define features and target variable
X = data.drop('condition', axis=1)  # Replace 'condition' with the exact column name
y = data['condition']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the processed data
pd.DataFrame(X_train).to_csv('X_train_classification.csv', index=False)
pd.DataFrame(X_test).to_csv('X_test_classification.csv', index=False)
pd.DataFrame(y_train).to_csv('y_train_classification.csv', index=False)
pd.DataFrame(y_test).to_csv('y_test_classification.csv', index=False)

print("\nData Preprocessing Complete. Processed files saved.")
