import streamlit as st
import joblib
import tensorflow as tf
import numpy as np

# -------------------------------
# Load model & scaler
# -------------------------------
loaded_scaler = joblib.load('scaler.joblib')
loaded_model = tf.keras.models.load_model('car_price_model.keras')

st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Car Price Prediction App")
st.write("Enter car details below to predict its price")

# -------------------------------
# User Inputs
# -------------------------------
year = st.number_input("Year", min_value=1990, max_value=2025, value=2018)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
owner = st.selectbox("Owner Type", [0, 1, 2, 3])

mileage = st.number_input("Mileage (km/l)", value=20.5)
engine = st.number_input("Engine (CC)", value=1200)
max_power = st.number_input("Max Power (bhp)", value=85)
seats = st.number_input("Seats", value=5)

torque_nm = st.number_input("Torque (Nm)", value=115)
max_torque_rpm = st.number_input("Max Torque RPM", value=3000)

fuel = st.selectbox("Fuel Type", ["Diesel", "LPG", "Petrol"])
seller_type = st.selectbox("Seller Type", ["Individual", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# -------------------------------
# One-Hot Encoding
# -------------------------------
fuel_Diesel = 1 if fuel == "Diesel" else 0
fuel_LPG = 1 if fuel == "LPG" else 0
fuel_Petrol = 1 if fuel == "Petrol" else 0

seller_type_Individual = 1 if seller_type == "Individual" else 0
seller_type_Trustmark = 1 if seller_type == "Trustmark Dealer" else 0

transmission_Manual = 1 if transmission == "Manual" else 0

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Price 💰"):

    new_car = np.array([[ 
        year, km_driven, owner, mileage, engine, max_power, seats,
        torque_nm, max_torque_rpm,
        fuel_Diesel, fuel_LPG, fuel_Petrol,
        seller_type_Individual, seller_type_Trustmark,
        transmission_Manual
    ]])

    # Scale input
    new_car_scaled = loaded_scaler.transform(new_car)

    # Predict (log value)
    prediction_log = loaded_model.predict(new_car_scaled)

    # Convert back
    final_price = np.expm1(prediction_log)

    st.success(f"💸 Predicted Price: ₹{final_price[0][0]:,.2f}")