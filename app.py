import streamlit as st
import numpy as np
import pickle

# Load model and encoders from your fixed folder
model_path = r"model.pkl"
encoder_path = r"encoders.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(encoder_path, "rb") as f:
    le_sex, le_class, le_embarked = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="🚢 Titanic Survival Predictor")
st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details to predict survival:")

# Input widgets
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 30)
fare = st.slider("Fare Paid", 0, 500, 50)
pclass = st.selectbox("Class", ["First", "Second", "Third"])

# Embarked port input — maps full name to dataset short code
embarked_display = {
    "Southampton": "S",
    "Cherbourg": "C",
    "Queenstown": "Q"
}
embarked_label = st.selectbox("Port of Embarkation", list(embarked_display.keys()))
embarked = embarked_display[embarked_label]

# Encode inputs
sex_enc = le_sex.transform([sex])[0]
pclass_enc = le_class.transform([pclass])[0]
embarked_enc = le_embarked.transform([embarked])[0]

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
