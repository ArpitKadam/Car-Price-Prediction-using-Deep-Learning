import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# Load the pre-trained model using pickle
with open('tensor_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to make predictions
def predict_price(features):
    features = np.array(features).reshape(1, -1)
    predicted_price = model.predict(features)
    return predicted_price[0][0]

# Streamlit app layout
st.title("Car Price Prediction App")

# Input fields for the features
st.header("Enter Car Details")
years = st.number_input("Years:", min_value=0, max_value=50, value=5)
km = st.number_input("Kilometers Driven:", min_value=0, value=50000)
rating = st.number_input("Rating (1-10):", min_value=1, max_value=10, value=5)
condition = st.number_input("Condition (1-10):", min_value=1, max_value=10, value=5)
economy = st.number_input("Economy (in km/l):", min_value=0, value=15)
top_speed = st.number_input("Top Speed (in km/h):", min_value=0, value=150)
hp = st.number_input("Horsepower:", min_value=0, value=100)
torque = st.number_input("Torque:", min_value=0, value=150)

# Button to predict
if st.button("Predict Price"):
    features = [years, km, rating, condition, economy, top_speed, hp, torque]
    predicted_price = predict_price(features)
    st.success(f"The predicted price of the car is: ${predicted_price:,.2f}")

# Optional: Add a footer or additional information
st.markdown("### Note:")
st.markdown("This model predicts the price of a car based on various features. Please enter the details accurately.")