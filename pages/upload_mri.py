
st.subheader("Patient Information")

patient_id = st.text_input("Patient ID")
patient_name = st.text_input("Patient Name")

age = st.number_input("Age", 1, 120)

gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

scan_date = st.date_input("Scan Date")

import pandas as pd

def save_patient_record(patient_data):

    df = pd.DataFrame([patient_data])

    df.to_csv(
        "database/patient_records.csv",
        mode="a",
        header=False,
        index=False
    )

record = {
    "patient_id": patient_id,
    "patient_name": patient_name,
    "age": age,
    "gender": gender,
    "prediction": predicted_class,
    "confidence": confidence
}

save_patient_record(record)
