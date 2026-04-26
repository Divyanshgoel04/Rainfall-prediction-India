# 🌧️ India Rainfall Predictor

A machine learning web app that predicts whether it will rain today in major Indian cities — built as a hands-on practice project to go beyond a guided Coursera course and turn it into something real.

**Live Demo →** [rainfall-prediction-india.streamlit.app](https://rainfall-prediction-india-gzg2fbrsymyx5x5nzy4lr3.streamlit.app)

---

## What this project is about

The app takes in weather conditions like temperature, humidity, wind speed, and air quality for cities like Mumbai, Delhi, Kolkata, Chennai, and Bengaluru — and predicts whether it will rain or not, along with a confidence score.

---

## How it works

The target variable (`RainToday`) is derived from the weather condition description — if the description contains words like "rain", "drizzle", "shower", or "thunder", it's labelled as a rainy day. This is then used to train a binary classifier.

The pipeline looks like this:

```
Raw Data → Feature Selection → Season Engineering → Train/Test Split
→ Preprocessing (Scaling + One-Hot Encoding) → Model Training → Prediction
```

---

## Models trained & results

| Model | Accuracy | Recall (Rain) |
|---|---|---|
| Random Forest | 95% | 0.58 |
| Logistic Regression | 92% | 0.50 |

Random Forest won on overall accuracy. Logistic Regression with `class_weight='balanced'` did better at catching actual rain days (higher recall), which matters more in a rainfall prediction context — missing a rainy day is worse than a false alarm.

The deployed app uses the **Random Forest** model.

---

## Features used for prediction

- City & Region
- Temperature & Feels Like temperature
- Humidity & Cloud cover
- Wind speed, gust speed & direction
- Pressure & Visibility
- UV Index
- Air quality (PM2.5, PM10, EPA Index)
- Indian Season (Summer / Monsoon / Post-Monsoon / Winter)

---

## Indian seasons (custom feature)

Unlike the original Australian project which used Southern Hemisphere seasons, this project uses India-specific seasons:

| Season | Months |
|---|---|
| Summer | March – May |
| Monsoon | June – September |
| Post-Monsoon | October – November |
| Winter | December – February |

---

## Dataset

**Indian Weather** — daily weather snapshots across 550+ Indian cities  
Source: [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/indian-weather-repository-daily-snapshot)  
Filtered to 5 major cities: Mumbai, Delhi, Kolkata, Chennai, Bengaluru

---

## Tech stack

- **Python** — core language
- **scikit-learn** — model training, preprocessing, GridSearchCV
- **pandas / numpy** — data manipulation
- **Streamlit** — web app UI
- **Streamlit Community Cloud** — free deployment

---

## Run it locally

```bash
git clone https://github.com/yourusername/rainfall-prediction-india.git
cd rainfall-prediction-india
pip install -r requirements.txt
streamlit run app.py
```

Make sure `IndianWeather.csv` is in the same folder as `app.py`.

---

## About

Built by **Divyansh Goel** — Computer Engineering undergrad at Thapar Institute of Engineering & Technology  
