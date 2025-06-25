# GDP Prediction Web App

A machine learning-powered web application that predicts **GDP per capita** based on various socio-economic and geographic inputs.

---

**Project link:** https://gdp-predictor-v0xf.onrender.com/

---

## 🚀 Features
- Clean, minimalistic UI with responsive design
- Prediction powered by a trained **Random Forest Regressor**
- Interactive input form for 16 real-world GDP indicators
- Built using **Flask**, **NumPy**, **scikit-learn**, and **HTML/CSS**

---

## 🧠 Model
The model was trained using the "Countries of the World" dataset and performs preprocessing such as:
- Label Encoding for categorical variables
- Scaling using `StandardScaler`
- Feature selection and IQR-based outlier removal

Final model: **Tuned RandomForestRegressor**

---

## 📦 Tech Stack
- Python 3.10+
- Flask
- NumPy
- Pandas
- scikit-learn
- Jinja2 (HTML templating)


---

## 💡 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/gdp-predictor.git
cd gdp-predictor

# 2. Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
