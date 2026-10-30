import streamlit as st
import joblib
from sklearn.datasets import load_wine

# Load the trained model
model = joblib.load("wine_model.joblib")

# Load dataset information
data = load_wine()

# Configure the web page
st.set_page_config(
    page_title="Wine Classification",
    page_icon="🍷"
)

# Application title
st.title("🍷 Wine Classification App")

st.write(
    "Enter the chemical measurements of a wine sample "
    "to predict its class."
)

# Create input fields
alcohol = st.number_input(
    "Alcohol",
    min_value=0.0,
    value=13.0
)

malic_acid = st.number_input(
    "Malic Acid",
    min_value=0.0,
    value=2.0
)

ash = st.number_input(
    "Ash",
    min_value=0.0,
    value=2.3
)

alcalinity = st.number_input(
    "Alcalinity of Ash",
    min_value=0.0,
    value=19.0
)

magnesium = st.number_input(
    "Magnesium",
    min_value=0.0,
    value=100.0
)

# Predict when the user clicks the button
if st.button("Predict Wine Class"):

    # Build an input row with all 13 features.
    # Unexposed features use sample values for this demo.
    sample = data.data[0].copy()

    sample[0] = alcohol
    sample[1] = malic_acid
    sample[2] = ash
    sample[3] = alcalinity
    sample[4] = magnesium

    prediction = model.predict([sample])

    predicted_class = data.target_names[prediction[0]]

    st.success("Predicted wine class: " + predicted_class)

    st.info(
        "This is an educational demonstration, "
        "not a wine quality or safety assessment."
    )