# 🌐 GDP per Capita Predictor (Flask App)

This Flask-based web application predicts a country's **GDP per Capita** using a trained `RandomForestRegressor` model. It takes 16 socio-economic and geographic input features and returns the expected GDP per person.

## 🔍 Features

- Interactive form UI to enter real-world country data
- Model trained on cleaned world data (`.csv`)
- Responsive, minimalistic UI
- Feature importance insights included

## 🧠 Tech Stack

- Python 3.x
- Flask
- Scikit-Learn (RandomForestRegressor)
- HTML/CSS (Vanilla)

## 🛠 How to Run Locally

```bash
pip install -r requirements.txt
python app.py
