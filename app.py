import streamlit as st
import numpy as np
import joblib

# Load models and scaler
scaler = joblib.load('models/base_feature_scaler.joblib')
models = {
    'Decision Tree': joblib.load('models/decision_tree_model.joblib'),
    'Random Forest': joblib.load('models/random_forest_model.joblib'),
    'SVM': joblib.load('models/svm_model.joblib'),
    'KNN': joblib.load('models/knn_model.joblib'),
    'Gradient Boosting': joblib.load('models/gradient_boosting_model.joblib')
}
encoder = joblib.load('models/label_encoder.joblib')

# Streamlit app layout
st.title('Crop Recommendation System')
st.markdown("""
This application provides recommendations for the most suitable crops based on environmental and soil conditions. 
Please adjust the input values in the sidebar to reflect your local conditions. Here's a brief explanation of each input:

- **Nitrogen (N), Phosphorus (P), Potassium (K):** Essential nutrients required by crops. Values should be in kg/ha.
- **Temperature (C):** The average temperature of the area in degrees Celsius.
- **Humidity (%):** Average relative humidity in percentage.
- **pH Level:** Soil acidity or alkalinity on a scale from 0 to 14.
- **Rainfall (mm):** Annual rainfall in millimeters.
- **Total Nutrients:** Sum of all nutrient inputs.
- **Temperature Humidity Index:** A combined index of temperature and humidity.
- **Log Rainfall:** The logarithmic value of rainfall, providing a transformed perspective of rainfall data.

Each model will provide a prediction based on these inputs.
""")

# Input fields for all 10 features
with st.sidebar:
    st.header('Input Features')
    N = st.number_input('Nitrogen (N)', min_value=0, max_value=200, value=50, help="Enter the amount of Nitrogen in the soil")
    P = st.number_input('Phosphorus (P)', min_value=0, max_value=200, value=40, help="Enter the amount of Phosphorus in the soil")
    K = st.number_input('Potassium (K)', min_value=0, max_value=200, value=30, help="Enter the amount of Potassium in the soil")
    with st.expander("Advanced Environmental Settings"):
        temperature = st.slider('Temperature (C)', -10.0, 50.0, 25.0)
        humidity = st.slider('Humidity (%)', 0.0, 100.0, 80.0)
        ph = st.slider('pH Level', 0.0, 14.0, 6.5)
        rainfall = st.slider('Rainfall (mm)', 0.0, 400.0, 100.0)
        total_nutrients = st.number_input('Total Nutrients', min_value=0, max_value=500, value=150)
        temperature_humidity = st.slider('Temperature Humidity Index', 0.0, 5000.0, 1500.0)
        log_rainfall = st.slider('Log Rainfall', 0.0, 10.0, 5.0)

# Create feature array and scale
features = np.array([[N, P, K, temperature, humidity, ph, rainfall, total_nutrients, temperature_humidity, log_rainfall]])
features_scaled = scaler.transform(features)

# Display predictions
st.header('Predictions')
for name, model in models.items():
    prediction = model.predict(features_scaled)
    crop = encoder.inverse_transform(prediction)[0]  # Decode prediction
    st.write(f'{name} predicts: {crop}')
