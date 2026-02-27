#  Credit Card Fraud Detection - Machine Learning Project


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 1 Load dataset
df = pd.read_csv("fraud.csv")
print("First 5 rows:\n", df.head())

print("\nColumns:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())

# 2 Check fraud distribution
plt.figure(figsize=(6,4))
sns.countplot(x=df["is_fraud"])
plt.title("Fraud Distribution (0 = Normal, 1 = Fraud)")
plt.show()

# 3 Features and target
X = df[["transaction_amount", "transaction_time", "customer_age",
        "merchant_id", "customer_location"]]
y = df["is_fraud"]

# 4 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# 5 Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6 Train model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train_scaled, y_train)

print("\n Model training completed!")

# 7 Evaluate model
y_pred = model.predict(X_test_scaled)

print("\n Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix:\n", cm)

# 8 Visualize confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 9 Predict a new transaction
new_transaction = pd.DataFrame({
    "transaction_amount": [900],
    "transaction_time": [23],
    "customer_age": [21],
    "merchant_id": [1050],
    "customer_location": [12]
})

new_scaled = scaler.transform(new_transaction)
prediction = model.predict(new_scaled)[0]

print("\n Prediction for new transaction (1=fraud, 0=normal):", prediction)
