import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model_path = "model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# App Title
st.title("🛒 E-Commerce Revenue Prediction")
st.write("Predict if a visitor will make a purchase based on browsing behavior.")

# Sidebar - User Inputs
st.sidebar.header("User Input Features")

# Numerical Inputs
def user_input():
    Administrative = st.sidebar.slider("Administrative Pages Visited", 0, 30, 3)
    Administrative_Duration = st.sidebar.slider("Time on Admin Pages (seconds)", 0, 2000, 100)
    Informational = st.sidebar.slider("Informational Pages Visited", 0, 30, 3)
    Informational_Duration = st.sidebar.slider("Time on Informational Pages (seconds)", 0, 2000, 100)
    ProductRelated = st.sidebar.slider("Product Pages Visited", 0, 1000, 10)
    ProductRelated_Duration = st.sidebar.slider("Time on Product Pages (seconds)", 0, 5000, 500)
    BounceRates = st.sidebar.slider("Bounce Rate (%)", 0.0, 1.0, 0.2)
    ExitRates = st.sidebar.slider("Exit Rate (%)", 0.0, 1.0, 0.3)
    PageValues = st.sidebar.slider("Page Value", 0.0, 100.0, 2.0)
    SpecialDay = st.sidebar.slider("Special Day (0-1)", 0.0, 1.0, 0.0)

    # Categorical Inputs (Dropdowns)
    Month = st.sidebar.selectbox("Month", ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    OperatingSystems = st.sidebar.selectbox("Operating System", [1, 2, 3, 4, 5, 6, 7, 8])
    Browser = st.sidebar.selectbox("Browser", [1, 2, 3, 4, 5, 6, 7, 8])
    Region = st.sidebar.selectbox("Region", list(range(1, 10)))
    TrafficType = st.sidebar.selectbox("Traffic Type", list(range(1, 21)))
    VisitorType = st.sidebar.selectbox("Visitor Type", ["Returning_Visitor", "New_Visitor"])
    Weekend = st.sidebar.selectbox("Weekend", [True, False])

    # Store in dictionary
    features = {
        "Administrative": Administrative,
        "Administrative_Duration": Administrative_Duration,
        "Informational": Informational,
        "Informational_Duration": Informational_Duration,
        "ProductRelated": ProductRelated,
        "ProductRelated_Duration": ProductRelated_Duration,
        "BounceRates": BounceRates,
        "ExitRates": ExitRates,
        "PageValues": PageValues,
        "SpecialDay": SpecialDay,
        "Month": Month,
        "OperatingSystems": OperatingSystems,
        "Browser": Browser,
        "Region": Region,
        "TrafficType": TrafficType,
        "VisitorType": VisitorType,
        "Weekend": Weekend
    }

    return pd.DataFrame([features])

# Get user input
input_df = user_input()

# One-Hot Encoding (to match training format)
input_df = pd.get_dummies(input_df)
expected_features = model.feature_names_in_

# Add missing columns (to match training data)
for col in expected_features:
    if col not in input_df.columns:
        input_df[col] = 0  # Add missing columns with 0

# Ensure column order matches model training
input_df = input_df[expected_features]

# Display user input
st.subheader("User Input Preview")
st.write(input_df)

# Predict revenue
if st.button("Predict Revenue"):
    prediction = model.predict(input_df)
    result = "💰 Purchase Expected!" if prediction[0] == 1 else "❌ No Purchase Expected."
    st.subheader("Prediction Result")
    st.markdown(f"## {result}")

# Footer
st.markdown("---")
st.markdown("Developed for **StatAlytical Showdown** 🚀")
