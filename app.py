import numpy as np
import streamlit as st
from src import predict, train

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")
st.write("Linear regression model: enter the house details and click Predict.")

area = st.number_input("Area (sq ft)", 300, 10000, 1800)
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 10, 2)
age = st.number_input("Age of house (years)", 0, 100, 5)
distance = st.number_input("Distance from city centre (km)", 0.0, 100.0, 8.0)
city = st.text_input("City", "Pune")

if st.button("Predict price"):
    row = [{"area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
            "age": age, "distance_city": distance, "city": city}]
    try:
        result = predict.predict(row)
    except Exception:
        with st.spinner("Preparing the model for the first time..."):
            train.main()
        result = predict.predict(row)
    st.success(f"Estimated price: {float(np.ravel(result)[0]):,.0f}")