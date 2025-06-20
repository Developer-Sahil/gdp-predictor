import os
from flask import *
import joblib
import pickle
import numpy as np

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "models/gdp_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
        
        input_values = [
            float(request.form.get(field))
            for field in ['Region', 'Population', 'Area_(sq._km)', 'Pop._Density_(per_sq._km)', 
               'Coastline_(coast/area_ratio)', 'Net_migration', 'Literacy_(%)', 
               'Phones_(per_1000)', 'Arable_(%)', 'Crops_(%)', 'Climate','Birthrate', 'Deathrate', 'Agriculture', 'Industry', 'Service']
        ]
        final_input = np.array(input_values).reshape(1, -1)
        scaled_data = scaler.transform(final_input)
       
        prediction = model.predict(scaled_data)[0]
        return render_template('gdp_pred.html', pred=round(prediction, 2))
       

if __name__ == '__main__':
    app.run(debug=True)
