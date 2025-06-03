import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Load and clean datasets ---
@st.cache_data
def load_data():
    symptoms_df = pd.read_csv("DiseaseAndSymptoms.csv")
    precautions_df = pd.read_csv("Disease_precaution.csv")
    return symptoms_df, precautions_df

def clean_symptom_data(df):
    symptom_cols = [col for col in df.columns if col.lower().startswith('symptom_')]

    def combine_symptoms(row):
        symptoms = []
        for col in symptom_cols:
            val = row[col]
            if pd.notna(val):
                val = str(val).lower().replace(' ', '')
                symptoms.append(val)
        return symptoms

    df['Symptoms'] = df.apply(combine_symptoms, axis=1)
    df = df.dropna(subset=['Disease'])
    return df[['Disease', 'Symptoms']]

# --- Build model ---
@st.cache_resource
def train_model(df):
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df['Symptoms'])
    y = df['Disease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, mlb

# --- Get precautions ---
def get_precautions(disease, precaution_df):
    precautions = precaution_df[precaution_df['Disease'].str.lower() == disease.lower()]
    if not precautions.empty:
        return precautions.iloc[0, 1:].dropna().tolist()
    return ["No precautions found for this disease."]

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Healthcare Diagnosis Assistant", layout="centered")

    # Inject CSS for font size and symptom text color
    st.markdown(
        """
        <style>
        /* Increase font size globally */
        html, body, [class*="css"]  {
            font-size: 18px;
        }
        /* Multiselect options text color */
        div[role="listbox"] span[data-baseweb="option"] {
            color: blue !important;
        }
        /* Multiselect selected items */
        div[class*="multiValue"] {
            color: blue !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🩺 Healthcare Diagnosis Assistant")
    st.write("Select your symptoms to get a possible disease prediction and suggested precautions.")

    symptoms_df, precaution_df = load_data()
    symptoms_df = clean_symptom_data(symptoms_df)
    model, mlb = train_model(symptoms_df)

    all_symptoms = sorted(set(symptom for sublist in symptoms_df['Symptoms'] for symptom in sublist))

    selected_symptoms = st.multiselect("Select Symptoms", options=all_symptoms)

    if st.button("Diagnose"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
            return

        user_input = [s.lower().replace(' ', '') for s in selected_symptoms]
        input_vector = mlb.transform([user_input])

        prediction = model.predict(input_vector)[0]
        st.success(f"🧾 Predicted Disease: **{prediction}**")

        precautions = get_precautions(prediction, precaution_df)
        st.markdown("### 💡 Suggested Precautions:")
        for i, p in enumerate(precautions, 1):
            st.markdown(f"- {p}")

if __name__ == "__main__":
    main()
