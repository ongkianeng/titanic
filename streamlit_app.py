# app.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
# This file contains the preprocessor and the model.
try:
    pipeline = joblib.load('titanic_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please run `train_model.py` first to generate it.")
    st.stop()

# Set up the title and a brief description
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="wide")
st.title('🚢 Titanic Survival Prediction')
st.write("""
This app predicts whether a passenger would have survived the Titanic disaster. 
Enter the passenger's details in the sidebar to see the prediction.
""")

# Create the user input interface in the sidebar
st.sidebar.header('Passenger Information')

# Passenger Class input
pclass = st.sidebar.selectbox('Passenger Class (Pclass)', (1, 2, 3))

# Sex input
sex = st.sidebar.selectbox('Sex', ('male', 'female'))

# Age input
age = st.sidebar.slider('Age', 0, 100, 29) # Default age set to the approximate mean

# SibSp (Number of Siblings/Spouses Aboard) input
sibsp = st.sidebar.slider('Number of Siblings/Spouses Aboard (SibSp)', 0, 8, 0)

# Parch (Number of Parents/Children Aboard) input
parch = st.sidebar.slider('Number of Parents/Children Aboard (Parch)', 0, 6, 0)

# Fare input
fare = st.sidebar.slider('Fare (in British Pounds)', 0.0, 513.0, 32.0)

# Embarked (Port of Embarkation) input
embarked = st.sidebar.selectbox('Port of Embarkation (Embarked)', ('C', 'Q', 'S'))

# Create a button to make a prediction
if st.sidebar.button('Predict Survival'):
    # Create a DataFrame from the user inputs
    # The feature names must match those used during training
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })

    
    # Apply SAME preprocessing as training
    input_data["Sex"] = input_data["Sex"].map({"male": 0, "female": 1})
    input_data["Embarked"] = input_data["Embarked"].map({"C": 0, "Q": 1, "S": 2})
    
    # Use the loaded pipeline to make a prediction
    prediction = pipeline.predict(input_data)[0]
    prediction_proba = pipeline.predict_proba(input_data)[0]

    # Display the results
    st.subheader('Prediction Result')
    
    if prediction == 1:
        st.success('**Survived** ✅')
        st.write(f"Probability of Survival: **{prediction_proba[1]:.2f}**")
    else:
        st.error('**Did Not Survive** ❌')
        st.write(f"Probability of Not Surviving: **{prediction_proba[0]:.2f}**")

    # Display the input data for reference
    st.write("---")
    st.subheader("Passenger Data Used for Prediction:")
    st.table(input_data)
