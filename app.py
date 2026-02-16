import streamlit as st
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 Intelligent House Price Prediction System")
st.markdown("### AI-Based Real Estate Price Estimation Platform")

st.write("Enter property details below:")

# -----------------------------
# User-friendly Inputs
# -----------------------------

monthly_income_rupees = st.number_input("💰 Monthly Family Income (₹)", min_value=5000.0, step=1000.0)
house_age = st.number_input("🏗️ House Age (years)", min_value=0.0, step=1.0)
num_rooms = st.number_input("🚪 Number of Rooms", min_value=1, step=1)
num_bedrooms = st.number_input("🛏️ Number of Bedrooms", min_value=1, step=1)
local_population = st.number_input("🏙️ Local Area Population", min_value=100.0, step=100.0)
people_per_house = st.number_input("👨‍👩‍👧 People per House", min_value=1.0, step=0.5)
latitude = st.number_input("🌍 Latitude", format="%.6f")
longitude = st.number_input("🌍 Longitude", format="%.6f")

# -----------------------------
# Data Conversion Logic
# -----------------------------
# Convert rupees → dataset scale (approx normalization logic)
# Dataset income is in 10k USD units → we normalize roughly
income_normalized = monthly_income_rupees / 100000  # simple normalization logic

# Build input vector in model order
input_data = np.array([[ 
    income_normalized,   # MedInc
    house_age,           # HouseAge
    num_rooms,           # AveRooms
    num_bedrooms,        # AveBedrms
    local_population,    # Population
    people_per_house,    # AveOccup
    latitude,            # Latitude
    longitude             # Longitude
]])

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict House Price"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    # Convert prediction to rupees (approx scaling for demo realism)
    predicted_price_rupees = prediction[0] * 1000000  # demo conversion scale

    st.success("🏡 Prediction Successful!")
    st.markdown(f"### 💰 Estimated House Price: ₹ {predicted_price_rupees:,.2f}")

    st.info("Prediction generated using Machine Learning (Random Forest Regression)")
    st.caption("This is an AI-based estimation for academic/demo purposes.")
