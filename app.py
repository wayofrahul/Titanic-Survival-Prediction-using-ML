import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("titanic_model.joblib")

model = load_model()

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Header Section
# ----------------------------
st.markdown(
    """
    <div style="text-align:center">
        <h1 style="color:#0077b6;">🚢 Titanic Survival Prediction App</h1>
        <p style="font-size:18px; color:#555;">Predict whether a passenger would have survived the Titanic tragedy based on their details.</p>
        <hr style="border:1px solid #0077b6">
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("🧾 Passenger Information")

pclass = st.sidebar.selectbox("Ticket Class (Pclass)", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 90, 25)
sibsp = st.sidebar.number_input("Siblings/Spouses aboard (SibSp)", 0, 10, 0)
parch = st.sidebar.number_input("Parents/Children aboard (Parch)", 0, 10, 0)
fare = st.sidebar.number_input("Fare", 0.0, 600.0, 32.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Derived features
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
title = "Mr" if sex == "male" else "Miss"

# ----------------------------
# Prediction Button
# ----------------------------
input_df = pd.DataFrame([{
    'Pclass': pclass,
    'Sex': sex,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked,
    'title': title,
    'FamilySize': family_size,
    'IsAlone': is_alone
}])

col1, col2, col3 = st.columns([1,2,1])

with col2:
    if st.button("🎯 Predict Survival"):
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.success(f"✅ The passenger is **likely to Survive** 🧍‍♀️\n\n**Survival Probability:** {proba:.2f}")
        else:
            st.error(f"❌ The passenger is **unlikely to Survive** ⚰️\n\n**Survival Probability:** {proba:.2f}")

# ----------------------------
# Footer
# ----------------------------
st.markdown(
    """
    <hr>
    <div style="text-align:center; color:gray;">
        <small>Developed By Rahul !! | Titanic Survival Prediction Using ML</small>
    </div>
    """,
    unsafe_allow_html=True
)

from streamlit_lottie import st_lottie
import requests

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ship = load_lottie("https://assets7.lottiefiles.com/packages/lf20_xlkxtmul.json")
st_lottie(lottie_ship, height=200)


st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(120deg, #d4fc79 0%, #96e6a1 100%);
    }
    </style>
    """,
    unsafe_allow_html=True
)
