import joblib
import streamlit as st
import pandas as pd
import os
import gdown

model_path = "final_model.pkl"

if not os.path.exists(model_path):
    url = "https://drive.google.com/file/d/14zj3Wf29w-iTyzOT4m4yNNNQCN0MzmED/view?usp=drive_link"
    gdown.download(url, model_path, quiet=False)

model = joblib.load(model_path)
st.title("Wildfire risk prediction model")
st.subheader("Enter weather data for the 3 days")

temps = [st.number_input(f"Temperature Day {i+1} (F)", value=0.0) for i in range(3)]
temperature_day_max = max(temps)
temperature_range_3days = max(temps) - min(temps)

winds = [st.number_input(f"Wind Speed Day {i+1} (mph or knots)",value= 0.0) for i in range(3)]
wind_speed_mean_3days = sum(winds) / len(winds)

rains = [st.number_input(f"Rainfall Day {i+1} (inches)", value=0.0) for i in range(3)]
precipitation_max_daily = max(rains)
rain_days_count = 0
for r in rains:
    if r > 0:
        rain_days_count = 1

input_data = pd.DataFrame([{
    "temperature_day_max": temperature_day_max,
    "temperature_range_3days": temperature_range_3days,
    "wind_speed_mean_3days": wind_speed_mean_3days,
    "precipitation_max_daily": precipitation_max_daily,
    "rain_days_count": rain_days_count
}])

if st.button("Predict Fire Risk"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"Fire risk detected! Probability: {prob:.2f}")
    else:
        st.success(f"No fire risk detected. Probability: {prob:.2f}")
