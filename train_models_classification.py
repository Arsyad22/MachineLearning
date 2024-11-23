import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, roc_auc_score, RocCurveDisplay
)
import matplotlib.pyplot as plt

# Load the preprocessed data
X_train = pd.read_csv('X_train_classification.csv')
X_test = pd.read_csv('X_test_classification.csv')
y_train = pd.read_csv('y_train_classification.csv').values.ravel()
y_test = pd.read_csv('y_test_classification.csv').values.ravel()

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# Evaluate Logistic Regression
print("\nLogistic Regression Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))
roc_auc_log = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])
print("ROC AUC Score:", roc_auc_log)

# Plot ROC Curve for Logistic Regression
plt.figure()
RocCurveDisplay.from_estimator(log_reg, X_test, y_test)
plt.title("ROC Curve: Logistic Regression")
plt.show()

# Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluate Random Forest
print("\nRandom Forest Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
roc_auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print("ROC AUC Score:", roc_auc_rf)

# Plot ROC Curve for Random Forest
plt.figure()
RocCurveDisplay.from_estimator(rf, X_test, y_test)
plt.title("ROC Curve: Random Forest")
plt.show()
