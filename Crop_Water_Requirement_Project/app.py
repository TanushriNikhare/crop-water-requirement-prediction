import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Crop Water Requirement Prediction",
    page_icon="🌱",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main application background */
    .stApp {
        background-color: #f5f9f4;
    }

    /* Main title */
    .main-title {
        text-align: center;
        color: #1b5e20;
        font-size: 38px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #555555;
        font-size: 18px;
        margin-bottom: 30px;
    }

    /* Section headings */
    .section-heading {
        color: #1b5e20;
        font-size: 25px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Input labels */
    label {
        font-weight: 600 !important;
    }

    /* Prediction metric */
    [data-testid="stMetric"] {
        background-color: white;
        border: 2px solid #66bb6a;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.08);
    }

    [data-testid="stMetricLabel"] {
        color: #2e7d32 !important;
        font-size: 18px !important;
        font-weight: bold !important;
    }

    [data-testid="stMetricValue"] {
        color: #1b5e20 !important;
        font-size: 38px !important;
        font-weight: bold !important;
    }

    /* Predict button */
    .stButton > button {
        width: 100%;
        background-color: #2e7d32;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 12px;
        font-size: 18px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #1b5e20;
        color: white;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #777777;
        font-size: 14px;
        margin-top: 35px;
        padding: 15px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "best_water_requirement_model.pkl"
    )

    features = joblib.load(
        "model_features.pkl"
    )

    return model, features


model, features = load_model()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🌱 Crop Water Requirement Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Prediction Using Weather, Soil and Field Parameters'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# PROJECT INTRODUCTION
# =========================================================

st.markdown(
    '<div class="section-heading">💧 About the Project</div>',
    unsafe_allow_html=True
)

st.write(
    "This application predicts the estimated daily water requirement "
    "of a crop using weather, soil and field parameters. "
    "The prediction is generated using a trained machine learning "
    "regression model."
)


# =========================================================
# INPUT PARAMETERS
# =========================================================

st.markdown(
    '<div class="section-heading">🌾 Enter Crop & Field Parameters</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


# =========================================================
# LEFT COLUMN
# =========================================================

with col1:

    temperature = st.number_input(
        "🌡️ Temperature (°C)",
        min_value=-10.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )

    humidity = st.number_input(
        "💦 Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=0.1
    )

    rainfall = st.number_input(
        "🌧️ Rainfall (mm/day)",
        min_value=0.0,
        value=2.0,
        step=0.1
    )

    soil_moisture = st.number_input(
        "🌱 Soil Moisture (%)",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.1
    )

    soil_ph = st.number_input(
        "🧪 Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=6.5,
        step=0.1
    )

    nitrogen = st.number_input(
        "🌿 Nitrogen (mg/kg)",
        min_value=0.0,
        value=50.0,
        step=1.0
    )


# =========================================================
# RIGHT COLUMN
# =========================================================

with col2:

    phosphorus = st.number_input(
        "🌿 Phosphorus (mg/kg)",
        min_value=0.0,
        value=30.0,
        step=1.0
    )

    potassium = st.number_input(
        "🌿 Potassium (mg/kg)",
        min_value=0.0,
        value=40.0,
        step=1.0
    )

    wind_speed = st.number_input(
        "💨 Wind Speed (m/s)",
        min_value=0.0,
        value=2.0,
        step=0.1
    )

    solar_radiation = st.number_input(
        "☀️ Solar Radiation (MJ/m²/day)",
        min_value=0.0,
        value=15.0,
        step=0.1
    )

    field_area = st.number_input(
        "🚜 Field Area (hectares)",
        min_value=0.01,
        value=1.0,
        step=0.1
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.write("")

predict_button = st.button(
    "💧 Predict Water Requirement"
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    input_data = pd.DataFrame({

        "temperature_c": [temperature],

        "humidity_percent": [humidity],

        "rainfall_mm_day": [rainfall],

        "soil_moisture_percent": [soil_moisture],

        "soil_ph": [soil_ph],

        "nitrogen_mg_kg": [nitrogen],

        "phosphorus_mg_kg": [phosphorus],

        "potassium_mg_kg": [potassium],

        "wind_speed_m_s": [wind_speed],

        "solar_radiation_mj_m2_day": [solar_radiation],

        "field_area_ha": [field_area]
    })


    # Keep exactly the same feature order
    # used during model training

    input_data = input_data[features]


    # Make prediction

    prediction = model.predict(input_data)[0]


    # =====================================================
    # RESULT
    # =====================================================

    st.success("Prediction completed successfully!")

    st.write("")

    st.metric(
        label="💧 Predicted Water Requirement",
        value=f"{prediction:.2f} mm/day"
    )

    st.write("")

    st.info(
        "The predicted value represents the estimated "
        "crop water requirement in millimetres per day."
    )


# =========================================================
# PROJECT HIGHLIGHTS
# =========================================================

st.divider()

st.markdown(
    '<div class="section-heading">📊 Project Highlights</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    st.subheader("🤖 Machine Learning")

    st.write(
        "Regression algorithms are used to predict "
        "crop water requirement."
    )


with col2:

    st.subheader("🌱 Input Parameters")

    st.write(
        "Weather, soil and field parameters "
        "are used as model inputs."
    )


with col3:

    st.subheader("💧 Prediction")

    st.write(
        "The estimated water requirement is "
        "displayed in mm/day."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        🌱 Crop Water Requirement Prediction |
        Machine Learning & Streamlit Project
    </div>
    """,
    unsafe_allow_html=True
)