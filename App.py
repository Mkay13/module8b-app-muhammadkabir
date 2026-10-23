import streamlit as st
import joblib
import numpy as np

# Load the trained model (saved from your notebook)
model = joblib.load("model.pkl")

# Page configuration
st.set_page_config(page_title="Iris Classifier", page_icon="🌸")

# Title and description
st.title("🌸 Iris Flower Classifier")
st.write("Adjust the sliders below to describe a flower, then click **Predict**.")

# Input widgets — four features matching the Iris dataset
sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1, step=0.1)
sepal_width  = st.slider("Sepal Width (cm)",  min_value=2.0, max_value=4.5, value=3.5, step=0.1)
petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4, step=0.1)
petal_width  = st.slider("Petal Width (cm)",  min_value=0.1, max_value=2.5, value=0.2, step=0.1)

# Predict button
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]

    # Map numeric prediction to class name (Iris dataset)
    class_names = ["Setosa", "Versicolor", "Virginica"]
    predicted_class = class_names[prediction]

    st.success(f"Predicted class: **{predicted_class}**")
    st.write(f"Raw prediction value: `{prediction}`")

# Optional: show the input summary
with st.expander("See input summary"):
    st.write(f"Sepal: {sepal_length} × {sepal_width} cm")
    st.write(f"Petal: {petal_length} × {petal_width} cm")