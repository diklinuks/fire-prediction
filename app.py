import joblib
import streamlit as st
import pandas as pd
import os
import gdown
import numpy as np

model_path = "final_model.pkl"

file_id = "14zj3Wf29w-iTyzOT4m4yNNNQCN0MzmED"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

if not os.path.exists(model_path):
    st.error("Model file not found")
    st.stop()

model = joblib.load(model_path)

st.title("Wildfire risk prediction model")
st.subheader("Enter weather data for the 3 days (average for day)")

temps = [st.number_input(f"Temperature Day {i+1} (°F)", value=0.0) for i in range(3)]
winds = [st.number_input(f"Wind Speed Day {i+1} (mph)", value=0.0) for i in range(3)]
rains = [st.number_input(f"Rainfall Day {i+1} (inches)", value=0.0) for i in range(3)]

def validate_inputs(values, label, min_val=0, max_val=999):
    for i, val in enumerate(values):
        if not np.isfinite(val):
            st.warning(f"{label} Day {i+1} must be a valid number.")
            return False
        if val < min_val or val > max_val:
            st.warning(f"{label} Day {i+1} is out of realistic range.")
            return False
    return True

valid_temp = validate_inputs(temps, "Temperature", -100, 200)
valid_wind = validate_inputs(winds, "Wind Speed", 0, 200)
valid_rain = validate_inputs(rains, "Rainfall", 0, 30)

if not (valid_temp and valid_wind and valid_rain):
    st.stop()

temperature_day_max = max(temps)
temperature_range_3days = max(temps) - min(temps)
wind_speed_mean_3days = sum(winds) / len(winds)
precipitation_max_daily = max(rains)
rain_days_count = int(any(r > 0 for r in rains))

input_data = pd.DataFrame([{
    "temperature_day_max": temperature_day_max,
    "temperature_range_3days": temperature_range_3days,
    "wind_speed_mean_3days": wind_speed_mean_3days,
    "precipitation_max_daily": precipitation_max_daily,
    "rain_days_count": rain_days_count
}])

if st.button("Predict Fire Risk"):
    try:
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"Fire risk detected! Probability: {prob:.2f}")
        else:
            st.success(f"No fire risk detected. Probability: {prob:.2f}")
    except Exception as e:
        st.exception(f"Prediction failed: {e}")
