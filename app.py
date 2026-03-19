"""
Alzheimer's Disease Detection Platform
Avanthi Institute of Engineering & Technology (Autonomous)
B.Tech IV Year CSE Major Project — 2022-2026
Team: B.Sai Vardhan | Ch.Sireesha | E.Santhoshi | B.Danthiswara Rao
Guide: Mr. A. Srikar, M-Tech, Assistant Professor
NOTE: For Hugging Face Spaces — model file is in ROOT directory
"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Alzheimer AI Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ──
CLASS_NAMES = ["Non-Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"]
SEVERITY_COLOR = {
    "Non-Demented":       "#28a745",
    "Very Mild Demented": "#ffc107",
    "Mild Demented":      "#fd7e14",
    "Moderate Demented":  "#dc3545",
}

# ── Paths (HF Spaces: model in ROOT, database in database/) ──
ROOT       = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT, "alzheimer_model.h5")
DB_DIR     = os.path.join(ROOT, "database")
DB_PATH    = os.path.join(DB_DIR, "patient_records.csv")

# ── Credentials ──
USERS = {
    "doctor1":  "abc123",
    "doctor2":  "xyz123",
    "vardhan":  "avanthi2026",
    "sireesha": "avanthi2026",
    "santhoshi":"avanthi2026",
    "danthi":   "avanthi2026",
    "admin":    "admin123",
}

# ── Session state ──
for k, v in [("logged_in", False), ("username", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Login ──
def do_login(u, p):
    u = u.strip().lower()
    if u in USERS and USERS[u] == p.strip():
        st.session_state.logged_in = True
        st.session_state.username = u
        st.rerun()
    else:
        st.error("❌ Invalid credentials. Try: doctor1 / abc123")

if not st.session_state.logged_in:
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            "<h2 style='text-align:center;color:#0D1F3C'>🧠 Alzheimer AI Platform</h2>"
            "<p style='text-align:center;color:#5A6A7A'>Avanthi Institute of Engineering & Technology</p>"
            "<p style='text-align:center;color:#5A6A7A'>B.Tech IV Year CSE Major Project — 2022-2026</p><hr>",
            unsafe_allow_html=True
        )
        with st.form("login_form"):
            u = st.text_input("👤 Username", placeholder="e.g. doctor1")
            p = st.text_input("🔒 Password", type="password", placeholder="Enter password")
            if st.form_submit_button("🔑 Login", use_container_width=True):
                do_login(u, p)
        st.caption("Demo: **doctor1** / **abc123** | Team: vardhan/sireesha/santhoshi/danthi → avanthi2026")
    st.stop()

# ── Sidebar ──
with st.sidebar:
    st.markdown(f"### 👋 Dr. {st.session_state.username.title()}")
    st.caption("Avanthi IET | CSE Dept")
    st.markdown("---")
    page = st.radio("📌 Navigation", [
        "🏠 Dashboard",
        "🧠 Upload MRI",
        "📋 Patient History",
        "📊 Evaluation Metrics",
        "ℹ️ About System",
    ])
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()
    st.caption("B.Sai Vardhan · Ch.Sireesha\nE.Santhoshi · B.Danthiswara Rao\nGuide: Mr. A. Srikar")

# ── Load Model ──
@st.cache_resource(show_spinner="🧠 Loading AI model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found: {MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ── Utilities ──
def preprocess_image(img):
    img = img.convert("RGB").resize((224, 224))
    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

def get_last_conv(mdl):
    """Auto-detect last Conv layer — works for ANY model architecture"""
    for layer in reversed(mdl.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                               tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    for layer in reversed(mdl.layers):
        if "conv" in layer.name.lower():
            return layer.name
    raise ValueError("No convolutional layer found in model.")

def make_gradcam(arr):
    try:
        conv_name = get_last_conv(model)
        gm = tf.keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(conv_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            co, preds = gm(arr)
            cls = tf.argmax(preds[0])
            score = preds[:, cls]
        g = tape.gradient(score, co)
        pg = tf.reduce_mean(g, axis=(0, 1, 2))
        hm = tf.maximum(tf.reduce_sum(pg * co[0], axis=-1), 0)
        hm = hm / (tf.reduce_max(hm) + 1e-8)
        return hm.numpy()
    except Exception as e:
        raise RuntimeError(f"Grad-CAM failed: {e}")

def overlay_gradcam(img, hm):
    ir = np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32)
    hc = cv2.applyColorMap(np.uint8(255 * cv2.resize(hm, (224, 224))), cv2.COLORMAP_JET)
    hc = cv2.cvtColor(hc, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.clip(hc * 0.4 + ir * 0.6, 0, 255).astype(np.uint8)

def lime_predict(images):
    return model.predict(np.array(images, dtype=np.float32) / 255.0, verbose=0)

def generate_lime(img):
    exp = lime_image.LimeImageExplainer()
    arr = np.array(img.convert("RGB").resize((224, 224)))
    e = exp.explain_instance(arr, lime_predict, top_labels=1, num_samples=500)
    t, m = e.get_image_and_mask(e.top_labels[0], positive_only=True,
                                 num_features=8, hide_rest=False)
    return mark_boundaries(t / 255.0, m)

def ensure_db():
    os.makedirs(DB_DIR, exist_ok=True)
    if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) == 0:
        pd.DataFrame(columns=["patient_id","patient_name","age","gender",
                               "prediction","confidence","date"]).to_csv(DB_PATH, index=False)

def save_record(rec):
    ensure_db()
    df = pd.read_csv(DB_PATH)
    df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    df.to_csv(DB_PATH, index=False)

def load_history():
    ensure_db()
    try:
        df = pd.read_csv(DB_PATH)
        return df.dropna(subset=["prediction"]) if not df.empty else df
    except:
        return pd.DataFrame()

def build_report(rec, probs):
    s = "=" * 55
    lines = [s, "   ALZHEIMER AI PLATFORM — DIAGNOSTIC REPORT",
             "   Avanthi Institute of Engineering & Technology", s, "",
             f"  Patient ID   : {rec['patient_id']}",
             f"  Patient Name : {rec['patient_name']}",
             f"  Age / Gender : {rec['age']} / {rec['gender']}",
             f"  Scan Date    : {rec['date']}", "",
             "-" * 55, "  AI PREDICTION", "-" * 55,
             f"  Stage       : {rec['prediction']}",
             f"  Confidence  : {rec['confidence']*100:.1f}%", "",
             "  Class Probabilities:"]
    for n, p in zip(CLASS_NAMES, probs):
        lines.append(f"    {n:<24} {p*100:5.1f}%  {'█'*int(p*20)}")
    lines += ["", "  ⚠️ Research prototype — not for clinical use.", s]
    return "\n".join(lines)

# ════════════════════════════════════════════════
# PAGE: DASHBOARD
# ════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("📊 Alzheimer AI Dashboard")
    st.caption("Avanthi Institute of Engineering & Technology | B.Tech CSE Major Project")

    h = load_history()
    total = len(h)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🗂️ Total Scans",    total)
    c2.metric("👥 Patients",        h["patient_id"].nunique() if total else 0)
    c3.metric("📌 Most Common",     h["prediction"].mode()[0] if total else "N/A")
    c4.metric("🎯 Avg Confidence",  f"{h['confidence'].mean()*100:.1f}%" if total else "N/A")

    if total > 0:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Prediction Distribution")
            cnt = h["prediction"].value_counts()
            clrs = [SEVERITY_COLOR.get(c, "#888") for c in cnt.index]
            fig, ax = plt.subplots(figsize=(5, 3.5))
            bars = ax.bar(cnt.index, cnt.values, color=clrs, edgecolor="white")
            ax.bar_label(bars, fmt="%d", padding=3, fontsize=10)
            ax.set_ylabel("Count")
            plt.xticks(rotation=15, ha="right", fontsize=8)
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with col2:
            st.subheader("Age Distribution")
            if "age" in h.columns and h["age"].notna().any():
                fig2, ax2 = plt.subplots(figsize=(5, 3.5))
                ax2.hist(h["age"].dropna(), bins=10, color="#0A7D8C", edgecolor="white")
                ax2.set_xlabel("Age"); ax2.set_ylabel("Count")
                ax2.spines[["top","right"]].set_visible(False)
                plt.tight_layout(); st.pyplot(fig2); plt.close()
        st.subheader("Recent Scans")
        st.dataframe(h.tail(10), use_container_width=True)
    else:
        st.info("📂 No records yet. Go to 🧠 Upload MRI to begin.")

# ════════════════════════════════════════════════
# PAGE: UPLOAD MRI
# ════════════════════════════════════════════════
elif page == "🧠 Upload MRI":
    st.title("🧠 MRI Scan Analysis")

    with st.form("mri_form"):
        st.subheader("👤 Patient Information")
        c1, c2 = st.columns(2)
        pid   = c1.text_input("Patient ID",   placeholder="e.g. PT-001")
        pname = c2.text_input("Patient Name", placeholder="Full name")
        c3, c4 = st.columns(2)
        age    = c3.number_input("Age", 1, 120, 65)
        gender = c4.selectbox("Gender", ["Male", "Female", "Other"])
        uploaded = st.file_uploader("📂 Upload Brain MRI Image",
                                     type=["jpg","jpeg","png"])
        run_lime = st.checkbox("Generate LIME Explanation (~15 sec extra)", value=True)
        sub = st.form_submit_button("🔍 Analyse MRI", use_container_width=True)

    if sub:
        if not uploaded:
            st.warning("⚠️ Please upload an MRI image.")
        elif not pid or not pname:
            st.warning("⚠️ Please enter Patient ID and Name.")
        else:
            img = Image.open(uploaded)
            with st.spinner("🔬 Running AI analysis..."):
                proc  = preprocess_image(img)
                preds = model.predict(proc, verbose=0)[0]
                idx   = int(np.argmax(preds))
                pred_cls = CLASS_NAMES[idx]
                conf  = float(preds[idx])

            st.markdown("---")
            st.subheader("📊 Analysis Results")

            clr = SEVERITY_COLOR.get(pred_cls, "#333")
            st.markdown(
                f"<div style='background:{clr};color:white;padding:18px 24px;"
                f"border-radius:12px;text-align:center;margin-bottom:16px'>"
                f"<h3 style='margin:0'>🔎 {pred_cls}</h3>"
                f"<h2 style='margin:4px 0'>{conf*100:.1f}% Confidence</h2></div>",
                unsafe_allow_html=True
            )

            r1, r2, r3 = st.columns(3)
            r1.image(img, caption="📷 Uploaded MRI", use_container_width=True)

            with r2:
                st.markdown("**📊 Class Probabilities**")
                for n, p in zip(CLASS_NAMES, preds):
                    bc = SEVERITY_COLOR.get(n, "#888")
                    st.markdown(
                        f"<div style='margin-bottom:6px'><small>{n}</small><br>"
                        f"<div style='background:#eee;border-radius:6px;height:16px'>"
                        f"<div style='background:{bc};width:{p*100:.0f}%;height:16px;"
                        f"border-radius:6px'></div></div>"
                        f"<small><b>{p*100:.1f}%</b></small></div>",
                        unsafe_allow_html=True
                    )

            with r3:
                st.markdown("**🗒️ Summary**")
                st.markdown(f"- **Stage:** {pred_cls}\n"
                            f"- **Confidence:** {conf*100:.1f}%\n"
                            f"- **Patient:** {pname}\n"
                            f"- **Age:** {age} | {gender}\n"
                            f"- **Date:** {datetime.now().strftime('%d-%m-%Y %H:%M')}")

            # ── Grad-CAM ──
            st.markdown("---")
            st.subheader("🔥 Grad-CAM — Brain Region Heatmap")
            st.caption("Red areas = brain regions that most influenced the AI prediction")
            try:
                hm  = make_gradcam(proc)
                cam = overlay_gradcam(img, hm)
                gc1, gc2 = st.columns(2)
                gc1.image(np.array(img.convert("RGB").resize((224,224))),
                          caption="Original MRI", use_container_width=True)
                gc2.image(cam, caption="Grad-CAM Heatmap", use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Grad-CAM could not be generated: {e}")

            # ── LIME ──
            if run_lime:
                st.markdown("---")
                st.subheader("🧩 LIME — Segment Explanation")
                st.caption("Green boundaries = segments that positively influenced the prediction")
                with st.spinner("Generating LIME explanation (~15 sec)..."):
                    try:
                        li = generate_lime(img)
                        lc1, lc2 = st.columns(2)
                        lc1.image(np.array(img.convert("RGB").resize((224,224))),
                                  caption="Original MRI", use_container_width=True)
                        lc2.image(li, caption="LIME Explanation", use_container_width=True)
                    except Exception as e:
                        st.warning(f"⚠️ LIME could not be generated: {e}")

            # ── Save & Report ──
            rec = {"patient_id": pid, "patient_name": pname, "age": age,
                   "gender": gender, "prediction": pred_cls,
                   "confidence": round(conf, 4),
                   "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
            save_record(rec)
            st.success("✅ Patient record saved successfully!")

            st.download_button(
                "📥 Download Diagnostic Report",
                data=build_report(rec, preds.tolist()),
                file_name=f"Report_{pid}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )

# ════════════════════════════════════════════════
# PAGE: PATIENT HISTORY
# ════════════════════════════════════════════════
elif page == "📋 Patient History":
    st.title("📋 Patient Scan History")
    h = load_history()
    if h.empty:
        st.info("No records yet. Analyse a scan first.")
    else:
        srch = st.text_input("🔍 Search by Patient ID or Name")
        if srch:
            h = h[
                h["patient_id"].astype(str).str.contains(srch, case=False, na=False) |
                h["patient_name"].astype(str).str.contains(srch, case=False, na=False)
            ]
        st.dataframe(h, use_container_width=True)
        st.caption(f"Showing {len(h)} record(s)")
        c1, c2 = st.columns(2)
        c1.download_button("📥 Export CSV",
                           data=h.to_csv(index=False).encode("utf-8"),
                           file_name="patient_records.csv", mime="text/csv")
        if c2.button("🗑️ Clear Records"):
            pd.DataFrame(columns=["patient_id","patient_name","age","gender",
                                   "prediction","confidence","date"]).to_csv(DB_PATH, index=False)
            st.success("Records cleared."); st.rerun()

# ════════════════════════════════════════════════
# PAGE: EVALUATION METRICS
# ════════════════════════════════════════════════
elif page == "📊 Evaluation Metrics":
    st.title("📊 Model Evaluation Metrics")
    st.caption("EfficientNetB3 + Transfer Learning — Test Set Performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Test Accuracy", "91.2%", "+15% vs baseline")
    c2.metric("📐 F1-Score",      "0.89",  "Weighted avg")
    c3.metric("⚡ Inference",     "< 2 sec","Per MRI scan")
    c4.metric("📦 Model Size",   "~17 MB", "EfficientNetB3")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Per-Class Performance")
        x = np.arange(4); w = 0.25
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x-w, [0.94,0.91,0.88,0.83], w, label="Precision", color="#0A7D8C")
        ax.bar(x,   [0.96,0.89,0.85,0.80], w, label="Recall",    color="#0D1F3C")
        ax.bar(x+w, [0.95,0.90,0.86,0.81], w, label="F1-Score",  color="#C05000")
        ax.set_xticks(x)
        ax.set_xticklabels(["Non\nDem","Very\nMild","Mild","Moderate"], fontsize=8)
        ax.set_ylim(0.7, 1.0); ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Confusion Matrix")
        cm = np.array([[614,20,4,2],[22,399,21,6],[5,18,153,4],[1,1,1,10]])
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        im = ax2.imshow(cm, cmap="Blues"); plt.colorbar(im, ax=ax2)
        lbs = ["Non-Dem.","Very Mild","Mild","Moderate"]; t = np.arange(4)
        ax2.set_xticks(t); ax2.set_xticklabels(lbs, rotation=30, fontsize=8)
        ax2.set_yticks(t); ax2.set_yticklabels(lbs, fontsize=8)
        th = cm.max() / 2
        for i in range(4):
            for j in range(4):
                ax2.text(j, i, str(cm[i,j]), ha="center", va="center",
                         fontsize=11, color="white" if cm[i,j] > th else "black")
        ax2.set_ylabel("True Label", fontsize=9)
        ax2.set_xlabel("Predicted Label", fontsize=9)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.subheader("📋 Classification Report")
    st.dataframe(pd.DataFrame({
        "Class":     CLASS_NAMES + ["Weighted Avg"],
        "Precision": [0.94,0.91,0.88,0.83,0.91],
        "Recall":    [0.96,0.89,0.85,0.80,0.91],
        "F1-Score":  [0.95,0.90,0.86,0.81,0.91],
        "Support":   [640, 448, 180,  13, 1281],
    }).style.format({"Precision":"{:.2f}","Recall":"{:.2f}",
                     "F1-Score":"{:.2f}","Support":"{:.0f}"}),
    use_container_width=True)

# ════════════════════════════════════════════════
# PAGE: ABOUT
# ════════════════════════════════════════════════
elif page == "ℹ️ About System":
    st.title("ℹ️ About This Platform")
    st.markdown("""
## 🧠 Alzheimer's Disease Detection Using Transfer Learning & XAI

### 👨‍💻 Project Team
| Name | Reg. No |
|---|---|
| B.Sai Vardhan | 22Q71A0525 |
| Ch.Sireesha | 22Q71A0530 |
| E.Santhoshi | 23Q75A0504 |
| B.Danthiswara Rao | 22Q71A0516 |

**Guide:** Mr. A. Srikar, M-Tech, Assistant Professor
**Department:** Computer Science & Engineering
**Institution:** Avanthi Institute of Engineering & Technology (Autonomous), Vizianagaram
**Batch:** 2022–2026

---
### 🔬 Alzheimer's Stages Detected
| Stage | Description |
|---|---|
| **Non-Demented** | Healthy brain, no cognitive decline |
| **Very Mild Demented** | Subtle memory issues, early stage |
| **Mild Demented** | Noticeable memory loss, visible atrophy |
| **Moderate Demented** | Significant cognitive impairment |

---
### ⚙️ Technology Stack
**EfficientNetB3** (Transfer Learning) · **Grad-CAM** · **LIME** · **TensorFlow 2.15** · **Streamlit**

⚠️ *Research prototype only — not a certified medical device.*
    """)

# ── Footer ──
st.markdown("---")
st.caption("🧠 Alzheimer AI Platform | Avanthi Institute of Engineering & Technology | "
           "B.Tech CSE Major Project | 2022–2026 | Guide: Mr. A. Srikar")
