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
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

from lime import lime_image
from skimage.segmentation import mark_boundaries

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Alzheimer MRI Analysis",
    page_icon="🧠",
    layout="wide"
)

# ======================================================
# SIDEBAR (Context + Dataset Info)
# ======================================================
with st.sidebar:
    st.markdown("## 🧠 Alzheimer MRI Analyzer")

    st.markdown(
        """
        **Purpose**  
        Academic AI system for analyzing brain MRI scans  

        **Approach**  
        Transfer Learning + Explainable AI (XAI)  

        **Explainability**  
        Grad-CAM and LIME  

        **Disclaimer**  
        This tool is for academic and research use only.  
        Not intended for clinical diagnosis.
        """
    )

    st.markdown("---")
    st.markdown("### 📊 Dataset Information")

    st.markdown(
        """
        **Dataset:** Alzheimer MRI (4 Classes)  
        **Source:** Public medical imaging dataset  

        **Classes:**  
        - Non-Demented  
        - Very Mild Demented  
        - Mild Demented  
        - Moderate Demented  

        **Note:**  
        A subset of the dataset was used for training and evaluation
        due to computational constraints.
        """
    )

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("alzheimer_model.h5")

model = load_model()

class_names = [
    "Non-Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented"
]

# ======================================================
# MAIN TITLE
# ======================================================
st.title("Explainable AI-Based Alzheimer’s Disease Detection")
st.write(
    "This application analyzes brain MRI images using deep learning "
    "and provides visual explanations to support transparency."
)

# ======================================================
# UPLOAD SECTION
# ======================================================
st.markdown("## 1️⃣ Upload Brain MRI")

st.info(
    "📌 Upload a **single axial brain MRI image** (JPG/PNG).\n\n"
    "Best results are obtained with clear, centered MRI slices."
)

uploaded_file = st.file_uploader(
    "Drag and drop MRI image here or click to browse",
    type=["jpg", "png", "jpeg"]
)

analyze_clicked = st.button("🔍 Analyze MRI", use_container_width=True)

# ======================================================
# IMAGE VALIDATION (Basic MRI Check)
# ======================================================
def is_likely_mri(img):
    img_gray = np.array(img.convert("L"))
    mean_intensity = img_gray.mean()
    return mean_intensity < 200  # MRI images are usually darker

# ======================================================
# IMAGE PREPROCESSING
# ======================================================
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ======================================================
# GRAD-CAM
# ======================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = tf.reduce_max(predictions, axis=1)

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

# ======================================================
# LIME (Stable Yellow Explanation)
# ======================================================
def lime_predict(images):
    images = np.array(images) / 255.0
    return model.predict(images)

def generate_lime_explanation(img):
    explainer = lime_image.LimeImageExplainer()

    img_rgb = img.convert("RGB").resize((224, 224))
    img_array = np.array(img_rgb)

    explanation = explainer.explain_instance(
        img_array,
        lime_predict,
        top_labels=1,
        hide_color=0,
        num_samples=300
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    lime_result = mark_boundaries(temp / 255.0, mask)
    return lime_result

# ======================================================
# MAIN LOGIC WITH LOADING SPINNER
# ======================================================
if uploaded_file is not None and analyze_clicked:
    with st.spinner("🧠 Analyzing MRI scan... Please wait"):
        img = Image.open(uploaded_file)

        if not is_likely_mri(img):
            st.error(
                "❌ The uploaded image does not appear to be a brain MRI.\n\n"
                "Please upload a valid axial brain MRI image."
            )
            st.stop()

        st.markdown("## 2️⃣ MRI Preview")
        st.image(img, use_column_width=True)

        processed_img = preprocess_image(img)

        preds = model.predict(processed_img)
        predicted_class = class_names[np.argmax(preds)]
        confidence = float(np.max(preds))

        # ---------------- Prediction Summary ----------------
        st.markdown("## 🧾 Prediction Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Alzheimer Stage", predicted_class)

        with col2:
            st.metric("Confidence Score", f"{confidence:.2f}")

        # ---------------- Probabilities ----------------
        with st.expander("📊 View Class Probabilities"):
            for i, cls in enumerate(class_names):
                st.progress(float(preds[0][i]))
                st.write(f"{cls}: {preds[0][i]:.2f}")

        # ---------------- Explainability ----------------
        st.markdown("## 🔍 Explainable AI Visualizations")

        tab1, tab2 = st.tabs(["Grad-CAM", "LIME"])

        with tab1:
            heatmap = make_gradcam_heatmap(processed_img, model)
            heatmap = cv2.resize(heatmap, (224, 224))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            img_rgb = img.convert("RGB").resize((224, 224))
            img_array = np.array(img_rgb).astype("float32")

            superimposed_img = heatmap * 0.7 + img_array * 0.6
            superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")

            st.image(
                superimposed_img,
                caption="Grad-CAM highlights regions influencing prediction",
                use_column_width=True
            )

        with tab2:
            lime_result = generate_lime_explanation(img)
            st.image(
                lime_result,
                caption="LIME shows locally influential regions",
                use_column_width=True
            )

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption(
    "© 2026 | Explainable AI for Alzheimer’s Disease | Academic Project"
)
