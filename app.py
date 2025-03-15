import streamlit as st
import numpy as np
import pickle

# Load the trained model & scaler
scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("titanic_model.pkl", "rb"))

# Streamlit UI
st.title("🚢 Titanic Survival Predictor")

# Input fields
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
fare = st.number_input("Fare ($)", min_value=0.0, max_value=500.0, value=30.0)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
sex = st.radio("Sex", ["Male", "Female"])
embarked = st.radio("Port of Embarkation", ["C", "Q", "S"])

# Encode inputs
sex_encoded = 1 if sex == "Male" else 0

# Embarked as one-hot encoding (3 features)
embarked_C = 1 if embarked == "C" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Convert input to model format (EXACTLY 8 features, matching training data)
passenger_data = np.array([[pclass, age, fare, sibsp, parch, sex_encoded, embarked_C, embarked_Q]])

# Scale the data
passenger_data = scaler.transform(passenger_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(passenger_data)
    result = "Survived ✨" if prediction[0] == 1 else "Did Not Survive 💀"
    st.success(result)


