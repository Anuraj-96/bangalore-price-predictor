import streamlit as st
#import pickle
import numpy as np
import json

# Load the trained model
import joblib
model = joblib.load('model.joblib')

# Load the scaler
scaler = joblib.load('scaler.joblib')

# Load the column names (feature list)
with open('columns.json', 'r') as f:
    data = json.load(f)
    data_columns = data["data_columns"]  

# Get location names (after 'total_sqft', 'bath', 'bhk')
locations = data_columns[3:]

# App title
st.set_page_config(page_title="Bangalore House Price Predictor", layout="centered")
st.title("🏠 Bangalore House Price Predictor")
st.markdown("Estimate the price of a house in Bangalore based on its features.")

# User inputs
sqft = st.number_input("Total Square Feet", min_value=300, step=50)
bath = st.selectbox("Number of Bathrooms", [1, 2, 3, 4, 5])
bhk = st.selectbox("Number of BHK", [2, 3, 4, 5])

location = st.selectbox("Location", sorted(locations))

# Predict button
if st.button("Predict Price"):
    try:
        # Create input vector with zeros
        input_data = np.zeros(len(data_columns))

        # Scale sqft, bath, bhk
        numeric_input = scaler.transform([[sqft, bath, bhk]])[0]
        input_data[0] = numeric_input[0]  # scaled sqft
        input_data[1] = numeric_input[1]  # scaled bath
        input_data[2] = numeric_input[2]  # scaled bhk
    

        # Set location one-hot
        if location in locations:
            loc_index = data_columns.index(location)
            input_data[loc_index] = 1

        # Predict
        prediction = model.predict([input_data])[0]
        st.success(f"💰 Estimated Price: ₹{round(prediction, 2)} Lakhs")

    except Exception as e:
        st.error(f"Error in prediction: {e}")