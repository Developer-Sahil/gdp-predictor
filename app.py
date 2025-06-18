from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model/gdp_model.pkl")

FEATURES = [
    "Phones_(per_1000)", "Agriculture", "Service", "Arable_(%)", "Crops_(%)",
    "Deathrate", "Net_migration", "Birthrate", "Area_(sq._mi.)", "Population",
    "Coastline_(coast/area_ratio)", "Literacy_(%)", "Region",
    "Pop._Density_(per_sq._mi.)", "Climate", "Industry"
]

@app.route('/')
def home():
    return render_template("index.html", features=FEATURES)

@app.route('/predict', methods=["POST"])
def predict():
    try:
        input_values = [float(request.form[feature]) for feature in FEATURES]
        prediction = model.predict([input_values])[0]
        return render_template("index.html", features=FEATURES, prediction=round(prediction, 2))
    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
