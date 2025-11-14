# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="HealthPulse - Obesity Predictor", layout="centered", page_icon=":green_heart:")

# Path to background image (change if needed)
BG_IMAGE_PATH = "/mnt/data/2b56d35f-61b1-4942-adc5-3cfdabc0c3a5.png"

# -------------------------
# HELPER: load and embed background image as CSS
# -------------------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    """
    Sets a full-page background image using base64 embedding.
    Adds some overlay dark gradient to keep text readable.
    """
    if not Path(png_file).exists():
        return
    bin_str = get_base64_of_bin_file(png_file)
    css = f"""
    <style>
    .stApp {{
      background-image: url("data:image/png;base64,{bin_str}");
      background-size: cover;
      background-position: center;
      background-attachment: fixed;
    }}
    /* dark overlay to make cards readable */
    .overlay {{
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.55);
      pointer-events: none;
      z-index: 0;
    }}
    /* translucent card */
    .card {{
      background: rgba(10,10,10,0.75);
      padding: 20px;
      border-radius: 16px;
      box-shadow: 0 6px 20px rgba(0,0,0,0.6);
      color: #e6f4ea;
    }}
    .accent {{
      color: #00e676;
    }}
    .stButton>button {
      background: linear-gradient(90deg,#00e676,#00c853);
      color: #000;
      font-weight: 600;
    }
    </style>
    <div class="overlay"></div>
    """
    st.markdown(css, unsafe_allow_html=True)

# Apply background
set_background(BG_IMAGE_PATH)

# -------------------------
# Load model + columns
# -------------------------
MODEL_PATH = "obesity_model.pkl"
COLUMNS_PATH = "model_columns.pkl"

model = None
model_columns = None
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.warning(f"Could not load model from {MODEL_PATH}. Prediction disabled. ({e})")

try:
    model_columns = joblib.load(COLUMNS_PATH)
except Exception as e:
    st.warning(f"Could not load model columns from {COLUMNS_PATH}. ({e})")

# -------------------------
# Page header (in a translucent card)
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h1 style='margin-bottom:6px'>HealthPulse</h1>", unsafe_allow_html=True)
st.markdown("<p style='margin-top:0;color:#bfeec9'>A friendly gateway to your healthier lifestyle — Obesity level predictor</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.write("")  # spacing

# -------------------------
# Input form inside another translucent card
# -------------------------
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("Enter details")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
        height_cm = st.number_input("Height (cm)", min_value=60.0, max_value=250.0, value=165.0, step=0.5)
        weight_kg = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=60.0, step=0.5)

    with col2:
        veg_freq = st.slider("Vegetable Consumption Frequency (1=low, 3=high)", 1, 3, 2)
        main_meals = st.slider("Number of Main Meals", 1, 5, 3)
        food_between = st.selectbox("Food Between Meals", ["No", "Sometimes", "Frequently", "Always"])
        water = st.slider("Daily Water Intake (L)", 0.0, 6.0, 2.0, 0.5)
        physical_act = st.slider("Physical Activity Frequency (0=none, 3=high)", 0, 3, 1)
        elec_time = st.slider("Device Usage (hours/day)", 0, 18, 3)
        transport = st.selectbox("Mode of Transportation", ["Walking", "Bike", "Public_Transport", "Automobile"])

    # Compute BMI consistently with training
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.write(f"**Computed BMI:** {bmi:.2f}")

    # Create input dataframe matching training column names (best effort)
    input_df = pd.DataFrame({
        "Gender": [gender],
        "Age": [age],
        "Height": [height_m],
        "Weight": [weight_kg],
        # match names used in your notebook / renaming map
        "Vegetable_Consumption_Frequency": [veg_freq],
        "Number_of_Main_Meals": [main_meals],
        "Food_Between_Meals": [food_between],
        "Daily_Water_Intake_Liters": [water],
        "Physical_Activity_Frequency": [physical_act],
        "Time_Using_Electronic_Devices": [elec_time],
        "Mode_of_Transportation": [transport],
        "BMI": [bmi]
    })

    # One-hot encode (drop_first used during training)
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Align to training columns
    if model_columns is not None:
        # model_columns might be an Index object; ensure list-like
        try:
            expected_cols = list(model_columns)
        except Exception:
            expected_cols = model_columns

        # Reindex to expected columns
        input_encoded = input_encoded.reindex(columns=expected_cols, fill_value=0)

    # Prediction
    if st.button("Predict Obesity Level"):
        if model is None or model_columns is None:
            st.error("Model or model columns not loaded. Ensure obesity_model.pkl and model_columns.pkl are in the app folder.")
        else:
            try:
                pred = model.predict(input_encoded)[0]
                st.success(f"Predicted obesity level: **{pred}**")
                # Optionally show probabilities if classifier supports predict_proba
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_encoded)[0]
                    classes = model.classes_
                    prob_df = pd.DataFrame({"class": classes, "probability": probs})
                    prob_df = prob_df.sort_values("probability", ascending=False).reset_index(drop=True)
                    st.write("Prediction probabilities:")
                    st.table(prob_df.style.format({"probability": "{:.2%}"}))
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Footer / optional dataset view
# -------------------------
st.write("")  # spacing
with st.expander("Show sample input that will be passed to model (after encoding)"):
    st.write("One-hot encoded features (aligned to training columns):")
    st.write(input_encoded)
