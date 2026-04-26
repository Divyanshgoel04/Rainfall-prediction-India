import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Rainfall Predictor",
    page_icon="🌧️",
    layout="centered"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f1923 0%, #1a2d3d 50%, #0f1923 100%);
    color: #e8f4f8;
}

.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}

.hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #4fc3f7, #81d4fa, #b3e5fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
    letter-spacing: -0.5px;
}

.hero p {
    color: #78909c;
    font-size: 0.95rem;
    font-weight: 300;
}

.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(79, 195, 247, 0.15);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}

.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4fc3f7;
    margin-bottom: 1rem;
    font-family: 'JetBrains Mono', monospace;
}

.result-rain {
    background: linear-gradient(135deg, rgba(13,71,161,0.6), rgba(21,101,192,0.4));
    border: 1px solid #1565c0;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}

.result-norain {
    background: linear-gradient(135deg, rgba(230,81,0,0.25), rgba(245,124,0,0.15));
    border: 1px solid #e65100;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}

.result-emoji {
    font-size: 4rem;
    display: block;
    margin-bottom: 0.5rem;
}

.result-text {
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.result-sub {
    font-size: 0.9rem;
    color: #b0bec5;
}

.confidence-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 99px;
    height: 8px;
    margin: 0.8rem 0 0.3rem;
    overflow: hidden;
}

.stSlider > div > div > div {
    color: #4fc3f7 !important;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    color: #b0bec5 !important;
    font-size: 0.85rem !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #0277bd, #0288d1);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem;
    font-size: 1rem;
    font-weight: 600;
    font-family: 'Sora', sans-serif;
    letter-spacing: 0.5px;
    cursor: pointer;
    transition: all 0.2s;
    margin-top: 0.5rem;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #01579b, #0277bd);
    transform: translateY(-1px);
}

.footer {
    text-align: center;
    color: #37474f;
    font-size: 0.75rem;
    margin-top: 3rem;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ── Load & train model ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on Indian weather data...")
def load_model():
    df = pd.read_csv("IndianWeather.csv")

    features = [
        'location_name', 'region', 'temperature_celsius', 'humidity',
        'wind_kph', 'wind_degree', 'pressure_mb', 'cloud',
        'feels_like_celsius', 'visibility_km', 'uv_index', 'gust_kph',
        'air_quality_PM2.5', 'air_quality_PM10', 'air_quality_us-epa-index',
        'condition_text', 'last_updated', 'precip_mm'
    ]
    df = df[features].dropna()

    rain_keywords = ['rain', 'drizzle', 'shower', 'thunder', 'sleet']
    df['RainToday'] = df['condition_text'].str.lower().apply(
        lambda x: 'Yes' if any(k in x for k in rain_keywords) else 'No'
    )

    def date_to_season(date):
        m = date.month
        if m in [3, 4, 5]:    return 'Summer'
        elif m in [6, 7, 8, 9]: return 'Monsoon'
        elif m in [10, 11]:   return 'Post-Monsoon'
        else:                  return 'Winter'

    df['last_updated'] = pd.to_datetime(df['last_updated'])
    df['Season'] = df['last_updated'].apply(date_to_season)
    df = df.drop(columns=['last_updated', 'condition_text', 'precip_mm'])

    X = df.drop(columns='RainToday')
    y = df['RainToday']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    num_features = X_train.select_dtypes(include=['number']).columns.tolist()
    cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('scaler', StandardScaler())]), num_features),
        ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100, max_depth=20,
            min_samples_split=2, random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    return model

# City → region mapping
CITY_REGION = {
    'Mumbai':    'Maharashtra',
    'Kolkata':   'West Bengal',
    'Chennai':   'Tamil Nadu',
    'Delhi':     'Delhi',
    'Bengaluru': 'Karnataka',
}

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🌧️ India Rainfall Predictor</h1>
    <p>Machine Learning · Indian Weather Repository · Random Forest</p>
</div>
""", unsafe_allow_html=True)

model = load_model()

# Location & Season
st.markdown('<div class="card"><div class="section-label">📍 Location & Time</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    city = st.selectbox("City", list(CITY_REGION.keys()))
with col2:
    season = st.selectbox("Season", ["Monsoon", "Summer", "Post-Monsoon", "Winter"])
st.markdown('</div>', unsafe_allow_html=True)

# Temperature & Humidity
st.markdown('<div class="card"><div class="section-label">🌡️ Temperature & Humidity</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("Temperature (°C)", 10.0, 48.0, 28.0, 0.5)
    feels_like = st.slider("Feels Like (°C)", 10.0, 55.0, 30.0, 0.5)
with col2:
    humidity = st.slider("Humidity (%)", 10, 100, 75)
    cloud = st.slider("Cloud Cover (%)", 0, 100, 50)
st.markdown('</div>', unsafe_allow_html=True)

# Wind & Visibility
st.markdown('<div class="card"><div class="section-label">💨 Wind & Visibility</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    wind_kph = st.slider("Wind Speed (kph)", 0.0, 80.0, 15.0, 0.5)
    gust_kph = st.slider("Gust Speed (kph)", 0.0, 100.0, 25.0, 0.5)
with col2:
    wind_degree = st.slider("Wind Direction (°)", 0, 360, 180)
    visibility_km = st.slider("Visibility (km)", 0.0, 10.0, 6.0, 0.1)
st.markdown('</div>', unsafe_allow_html=True)

# Pressure & Air Quality
st.markdown('<div class="card"><div class="section-label">🌫️ Pressure & Air Quality</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    pressure_mb = st.slider("Pressure (mb)", 980.0, 1030.0, 1008.0, 0.5)
    uv_index = st.slider("UV Index", 0.0, 12.0, 3.0, 0.5)
with col2:
    pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 350.0, 40.0, 1.0)
    pm10 = st.slider("PM10 (µg/m³)", 0.0, 400.0, 60.0, 1.0)

epa_index = st.select_slider(
    "US EPA Air Quality Index",
    options=[1, 2, 3, 4, 5, 6],
    value=2,
    format_func=lambda x: {1:"Good",2:"Moderate",3:"Unhealthy (Sensitive)",4:"Unhealthy",5:"Very Unhealthy",6:"Hazardous"}[x]
)
st.markdown('</div>', unsafe_allow_html=True)

# Predict button
if st.button("🔍 Predict Rainfall"):
    input_df = pd.DataFrame([{
        'location_name':            city,
        'region':                   CITY_REGION[city],
        'temperature_celsius':      temperature,
        'humidity':                 humidity,
        'wind_kph':                 wind_kph,
        'wind_degree':              wind_degree,
        'pressure_mb':              pressure_mb,
        'cloud':                    cloud,
        'feels_like_celsius':       feels_like,
        'visibility_km':            visibility_km,
        'uv_index':                 uv_index,
        'gust_kph':                 gust_kph,
        'air_quality_PM2.5':        pm25,
        'air_quality_PM10':         pm10,
        'air_quality_us-epa-index': epa_index,
        'Season':                   season,
    }])

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    classes = model.classes_
    yes_idx = list(classes).index('Yes')
    confidence = proba[yes_idx] if prediction == 'Yes' else proba[1 - yes_idx]

    if prediction == 'Yes':
        st.markdown(f"""
        <div class="result-rain">
            <span class="result-emoji">🌧️</span>
            <div class="result-text">Rain Expected</div>
            <div class="result-sub">The model predicts rainfall for {city} today</div>
            <div class="confidence-bar-bg">
                <div style="width:{proba[yes_idx]*100:.0f}%;background:linear-gradient(90deg,#1565c0,#42a5f5);height:8px;border-radius:99px;"></div>
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#90caf9;">
                Rain probability: {proba[yes_idx]*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-norain">
            <span class="result-emoji">☀️</span>
            <div class="result-text">No Rain Expected</div>
            <div class="result-sub">Conditions look clear for {city} today</div>
            <div class="confidence-bar-bg">
                <div style="width:{proba[1-yes_idx]*100:.0f}%;background:linear-gradient(90deg,#e65100,#ff9800);height:8px;border-radius:99px;"></div>
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#ffcc80;">
                No-rain probability: {proba[1-yes_idx]*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Built with Streamlit · Indian Weather Repository (2023–2024) · Random Forest Classifier
</div>
""", unsafe_allow_html=True)
