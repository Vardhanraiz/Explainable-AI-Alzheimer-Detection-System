import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt

from lime import lime_image
from skimage.segmentation import mark_boundaries

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Alzheimer AI Platform",
    layout="wide"
)

# --------------------------------------------------
# LOGIN SYSTEM
# --------------------------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    if username == "doctor1" and password == "1234":
        st.session_state.logged_in = True
    else:
        st.error("Invalid username or password")

if not st.session_state.logged_in:

    st.title("Alzheimer AI Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    st.button("Login", on_click=login)

    st.stop()

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Upload MRI",
        "Patient History",
        "About System"
    ]
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/alzheimer_model.h5")

model = load_model()

class_names = [
    "Non-Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented"
]

# --------------------------------------------------
# PREPROCESS IMAGE
# --------------------------------------------------

def preprocess_image(img):

    img = img.convert("RGB")
    img = img.resize((224,224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# --------------------------------------------------
# GRAD-CAM
# --------------------------------------------------

def make_gradcam_heatmap(img_array, model):

    last_conv_layer = model.get_layer("top_conv")

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = tf.reduce_max(predictions, axis=1)

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

# --------------------------------------------------
# LIME
# --------------------------------------------------

def lime_predict(images):

    images = np.array(images) / 255.0

    return model.predict(images)

def generate_lime(img):

    explainer = lime_image.LimeImageExplainer()

    img_rgb = img.convert("RGB").resize((224,224))

    img_array = np.array(img_rgb)

    explanation = explainer.explain_instance(
        img_array,
        lime_predict,
        top_labels=1,
        num_samples=300
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5
    )

    return mark_boundaries(temp/255.0, mask)

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.title("Alzheimer AI Dashboard")

    try:

        history = pd.read_csv("database/patient_records.csv")

        total_scans = len(history)

        total_patients = history["patient_id"].nunique()

        most_common = history["prediction"].mode()[0]

    except:

        total_scans = 0
        total_patients = 0
        most_common = "N/A"

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Scans", total_scans)
    col2.metric("Total Patients", total_patients)
    col3.metric("Most Common Stage", most_common)

    st.info("AI platform for Alzheimer detection from MRI images")

# --------------------------------------------------
# UPLOAD MRI
# --------------------------------------------------

if page == "Upload MRI":

    st.title("Upload Brain MRI")

    st.subheader("Patient Information")

    patient_id = st.text_input("Patient ID")
    patient_name = st.text_input("Patient Name")

    age = st.number_input("Age", 1, 120)

    gender = st.selectbox(
        "Gender",
        ["Male","Female","Other"]
    )

    uploaded_file = st.file_uploader(
        "Upload MRI image",
        type=["jpg","png","jpeg"]
    )

    analyze = st.button("Analyze MRI")

    if uploaded_file and analyze:

        img = Image.open(uploaded_file)

        st.image(img)

        processed = preprocess_image(img)

        preds = model.predict(processed)

        predicted_class = class_names[np.argmax(preds)]

        confidence = float(np.max(preds))

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        col1.metric("Stage", predicted_class)
        col2.metric("Confidence", f"{confidence:.2f}")

        # probability chart

        fig, ax = plt.subplots()

        ax.bar(class_names, preds[0])

        st.pyplot(fig)

        # save patient record

        record = {

            "patient_id":patient_id,
            "patient_name":patient_name,
            "age":age,
            "gender":gender,
            "prediction":predicted_class,
            "confidence":confidence,
            "date":datetime.now()

        }

        df = pd.DataFrame([record])

        df.to_csv(
            "database/patient_records.csv",
            mode="a",
            header=False,
            index=False
        )

        # GradCAM

        st.subheader("Grad-CAM")

        heatmap = make_gradcam_heatmap(processed, model)

        heatmap = cv2.resize(heatmap,(224,224))

        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)

        img_rgb = img.resize((224,224))

        img_array = np.array(img_rgb).astype("float32")

        result = heatmap*0.7 + img_array*0.6

        result = np.clip(result,0,255).astype("uint8")

        st.image(result)

        # LIME

        st.subheader("LIME Explanation")

        lime_img = generate_lime(img)

        st.image(lime_img)

        # report

        report = f"""

Patient ID: {patient_id}

Patient Name: {patient_name}

Prediction: {predicted_class}

Confidence: {confidence}

"""

        st.download_button(
            "Download MRI Report",
            report,
            file_name="report.txt"
        )

# --------------------------------------------------
# PATIENT HISTORY
# --------------------------------------------------

if page == "Patient History":

    st.title("Patient Scan History")

    try:

        history = pd.read_csv("database/patient_records.csv")

        st.dataframe(history)

    except:

        st.warning("No patient records yet.")

# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

if page == "About System":

    st.title("About")

    st.write(
        """
This system detects Alzheimer’s disease stages
using deep learning and explainable AI.

Model: Transfer Learning CNN

Explainability: Grad-CAM and LIME
"""
    )
