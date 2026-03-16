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

elif authentication_status:

    authenticator.logout("Logout", "sidebar")

    st.sidebar.success(f"Welcome {name}")

    st.title("Alzheimer AI System Dashboard")
