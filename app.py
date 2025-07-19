import streamlit as st
import numpy as np
import pickle

# Load model and encoders
model_path = "model.pkl"
encoder_path = "encoders.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(encoder_path, "rb") as f:
    le_sex, le_class, le_embarked = pickle.load(f)

st.set_page_config(page_title="🚢 Titanic Survival Predictor")

st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details to predict survival:")

# Input widgets
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 30)
fare = st.slider("Fare Paid", 0, 500, 50)
pclass = st.selectbox("Class", ["First", "Second", "Third"])
embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])

# Encode inputs
try:
    sex_enc = le_sex.transform([sex])[0]
    pclass_enc = le_class.transform([pclass])[0]
    embarked_enc = le_embarked.transform([embarked])[0]
except ValueError:
    st.error("⚠️ Input label not recognized by encoder. Check your training data encodings.")

# Predict
input_data = np.array([[sex_enc, age, fare, pclass_enc, embarked_enc]])
prediction = model.predict(input_data)[0]
proba = model.predict_proba(input_data)[0]

# Output
st.subheader("Prediction")
if prediction == 1:
    st.success("🎉 The passenger would have **survived**.")
else:
    st.error("💀 The passenger would **not** have survived.")

st.subheader("Prediction Probability")
st.write(f"Survival: {proba[1]*100:.2f}%")
st.write(f"Death: {proba[0]*100:.2f}%")
