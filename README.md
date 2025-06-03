# 🩺 Healthcare Diagnosis Assistant

This is a Streamlit-based web application that predicts potential diseases based on user-selected symptoms and suggests corresponding health precautions.

## 🚀 Demo

👉 [Live App on Streamlit Cloud](https://healthcare-diagnosis-assistant-nd6exsduhksa5mtoozbhuh.streamlit.app)

---

## 🧠 Features

- Input multiple symptoms via an interactive UI
- Predict disease using a trained `RandomForestClassifier`
- Recommend health precautions from a mapped dataset
- Clean and minimal interface built with Streamlit

---

## 📁 Dataset Sources

- `DiseaseAndSymptoms.csv`: Contains diseases and their associated symptoms.
- `Disease_precaution.csv`: Maps diseases to suggested precautions.

---

## 🧰 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, Pandas, scikit-learn
- **ML Model**: Random Forest Classifier
- **Deployment**: Streamlit Cloud

---

## 🛠 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/anushka-0907/healthcare-diagnosis-assistant.git
cd healthcare-diagnosis-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
