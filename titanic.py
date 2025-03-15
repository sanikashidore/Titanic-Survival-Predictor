import pandas as pd  
from sklearn.metrics import accuracy_score  


# Load dataset
df = pd.read_csv("train.csv")  

# Show first 5 rows
#print(df.head())
#print(df.tail())  

# Check missing values
#print(df.isnull().sum())  

df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

df = df.copy()  # Avoids warning

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

#--------------------------------------------------


X = df.drop(columns=["Survived"])  # Features  
y = df["Survived"]  # Target  

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#--------------------------------------------------
#before training
from sklearn.preprocessing import StandardScaler  

scaler = StandardScaler()  
X_train = scaler.fit_transform(X_train)  
X_test = scaler.transform(X_test)  


#--------------------------------------------------
#training
from sklearn.linear_model import LogisticRegression  

model = LogisticRegression(max_iter=1000)  # More iterations
model.fit(X_train, y_train)  # Train the model  
  
y_pred = model.predict(X_test)  
accuracy = accuracy_score(y_test, y_pred)  
print(f"Accuracy: {accuracy:.2f}")  


#--------------------------------------------------
#after training
import numpy as np

# Test with an example passenger
test_passenger = np.array([[1, 22, 7.25, 1, 0, 0, 0, 1]])  # Adjust values

# Scale it
test_passenger = scaler.transform(test_passenger)

# Predict
print("Prediction:", model.predict(test_passenger))  # Should print 1 or 0

 




#--------------------------------------------------
# Save model & scaler
import pickle

# Save the trained model
with open("titanic_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save the scaler (if you're using one)
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("✅ Model saved as titanic_model.pkl")
print("✅ Scaler saved as scaler.pkl")


#--------------------------------------------------

# Load test.csv
df_test = pd.read_csv("test.csv")

# Drop unnecessary columns
df_test.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

# Fill missing values
df_test["Age"] = df_test["Age"].fillna(df_test["Age"].median())
df_test["Fare"] = df_test["Fare"].fillna(df_test["Fare"].median())  # Important!
df_test["Embarked"] = df_test["Embarked"].fillna(df_test["Embarked"].mode()[0])

# One-hot encoding for categorical features
df_test = pd.get_dummies(df_test, columns=["Sex", "Embarked"], drop_first=True)

# Ensure columns match training data
expected_columns = X.columns  # Use same columns as train.csv
df_test = df_test.reindex(columns=expected_columns, fill_value=0)

# Scale features using the same scaler
X_test_real = scaler.transform(df_test)

# Predict on real test set
test_predictions = model.predict(X_test_real)

# Count predictions
print("Survived (1):", sum(test_predictions == 1))
print("Did Not Survive (0):", sum(test_predictions == 0))