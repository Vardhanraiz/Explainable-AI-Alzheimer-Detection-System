
import pandas as pd
import streamlit as st

st.title("Patient Scan History")

history = pd.read_csv("database/patient_records.csv")

st.dataframe(history)
