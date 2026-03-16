import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

with open("database/users.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"]
)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username or password is incorrect")

elif authentication_status == None:
    st.warning("Please enter your username and password")

st.set_page_config(
    page_title="Alzheimer AI Platform",
    layout="wide"
)


elif authentication_status:

    authenticator.logout("Logout", "sidebar")

    st.sidebar.success(f"Welcome {name}")

    st.title("Alzheimer AI System Dashboard")


st.set_page_config(
    page_title="Alzheimer AI Platform",
    layout="wide"
)
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Upload MRI",
        "Patient History",
        "About System"
    ]
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Scans", 120)

with col2:
    st.metric("Patients", 60)

with col3:
    st.metric("Model Accuracy", "74%")

import matplotlib.pyplot as plt

labels = ["Non", "Very Mild", "Mild", "Moderate"]

plt.bar(labels, preds[0])

st.pyplot(plt)
