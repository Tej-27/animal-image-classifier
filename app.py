import streamlit as st
import numpy as np
from PIL import Image
import joblib
from textwrap import dedent


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Animal Vision",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(dedent("""
<style>

.stApp {
    background: #0f172a;
    color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.hero {
    text-align: center;
    padding: 35px 20px 30px 20px;
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 19px;
    color: #94a3b8;
    margin-bottom: 8px;
}

.hero-description {
    font-size: 15px;
    color: #64748b;
}

.card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 18px;
    padding: 25px;
    margin-bottom: 20px;
}

.card-title {
    font-size: 20px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 15px;
}

.prediction-card {
    background: linear-gradient(
        135deg,
        #1e293b,
        #172554
    );

    border: 1px solid #3b82f6;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    margin-top: 10px;
}

.prediction-label {
    color: #94a3b8;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.prediction-name {
    font-size: 40px;
    font-weight: 800;
    color: #60a5fa;
    margin: 12px 0;
}

.confidence {
    font-size: 18px;
    color: #cbd5e1;
}

.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 22px;
    text-align: center;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    color: #60a5fa;
}

.metric-label {
    color: #94a3b8;
    font-size: 14px;
    margin-top: 5px;
}

.info-box {
    background: #172033;
    border-left: 4px solid #3b82f6;
    padding: 15px 18px;
    border-radius: 8px;
    color: #cbd5e1;
    margin-top: 15px;
}

.footer {
    text-align: center;
    padding: 35px 0 10px 0;
    color: #64748b;
    font-size: 13px;
}

[data-testid="stFileUploader"] {
    background: #1e293b;
    border: 2px dashed #475569;
    border-radius: 18px;
    padding: 15px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid #475569;
    padding: 10px;
    font-weight: 600;
}

section[data-testid="stSidebar"] {
    background: #020617;
}

</style>
"""), unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("animal_classifier_compressed.pkl")


model = load_model()

classes = ["cat", "dog", "wild"]


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# 🐾 AI Animal Vision")

    st.markdown("---")

    st.markdown("## About the Model")

    st.write(
        "This application uses a Machine Learning model "
        "to classify animal images into three categories."
    )

    st.markdown("## Supported Classes")

    st.markdown("""
🐱 **Cat**

🐶 **Dog**

🦁 **Wild**
""")

    st.markdown("---")

    st.markdown("## Model Details")

    st.write("**Algorithm:** Random Forest")
    st.write("**Input:** 64 × 64 RGB image")
    st.write("**Features:** 12,288")
    st.write("**Trees:** 300")
    st.write("**Validation Accuracy:** 82.60%")

    st.markdown("---")

    st.info(
        "For best results, upload a clear image "
        "containing one main animal."
    )


# =========================================================
# HERO SECTION
# =========================================================

st.markdown(dedent("""
<div class="hero">

<div class="hero-title">
🐾 AI Animal Vision
</div>

<div class="hero-subtitle">
Intelligent Animal Image Classification
</div>

<div class="hero-description">
Upload an image and let machine learning identify
whether it contains a Cat, Dog, or Wild animal.
</div>

</div>
"""), unsafe_allow_html=True)


# =========================================================
# METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(dedent("""
    <div class="metric-card">
    <div class="metric-value">82.6%</div>
    <div class="metric-label">Validation Accuracy</div>
    </div>
    """), unsafe_allow_html=True)

with col2:
    st.markdown(dedent("""
    <div class="metric-card">
    <div class="metric-value">3</div>
    <div class="metric-label">Animal Classes</div>
    </div>
    """), unsafe_allow_html=True)

with col3:
    st.markdown(dedent("""
    <div class="metric-card">
    <div class="metric-value">300</div>
    <div class="metric-label">Random Forest Trees</div>
    </div>
    """), unsafe_allow_html=True)

with col4:
    st.markdown(dedent("""
    <div class="metric-card">
    <div class="metric-value">64×64</div>
    <div class="metric-label">Input Resolution</div>
    </div>
    """), unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(dedent("""
<div class="card">

<div class="card-title">
📤 Upload an Animal Image
</div>

</div>
"""), unsafe_allow_html=True)


uploaded_file = st.file_uploader(
    "Drag and drop an image here",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG and WEBP"
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    try:

        # -------------------------------------------------
        # LOAD IMAGE
        # -------------------------------------------------

        image = Image.open(uploaded_file).convert("RGB")


        # -------------------------------------------------
        # IMAGE + RESULT COLUMNS
        # -------------------------------------------------

        image_col, prediction_col = st.columns(
            [1, 1],
            gap="large"
        )


        # -------------------------------------------------
        # IMAGE PREVIEW
        # -------------------------------------------------

        with image_col:

            st.markdown(
                "### 🖼️ Uploaded Image"
            )

            st.image(
                image,
                use_container_width=True
            )

            st.caption(
                f"Original image size: "
                f"{image.width} × {image.height} pixels"
            )


        # -------------------------------------------------
        # PREPROCESSING
        # -------------------------------------------------

        image_resized = image.resize((64, 64))

        image_array = np.array(image_resized)

        image_flat = image_array.reshape(1, -1)

        image_flat = (
            image_flat.astype("float32") / 255.0
        )


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        prediction = model.predict(image_flat)[0]

        predicted_class = classes[prediction]


        # -------------------------------------------------
        # PROBABILITIES
        # -------------------------------------------------

        probabilities = model.predict_proba(
            image_flat
        )[0]

        confidence = (
            probabilities[prediction] * 100
        )


        # -------------------------------------------------
        # PREDICTION CARD
        # -------------------------------------------------

        animal_emoji = {
            "cat": "🐱",
            "dog": "🐶",
            "wild": "🦁"
        }

        display_name = predicted_class.upper()

        with prediction_col:

            st.markdown(dedent(f"""
            <div class="prediction-card">

            <div class="prediction-label">
            AI Prediction
            </div>

            <div class="prediction-name">
            {animal_emoji[predicted_class]} {display_name}
            </div>

            <div class="confidence">
            Confidence: <strong>{confidence:.2f}%</strong>
            </div>

            </div>
            """), unsafe_allow_html=True)


            st.markdown("### 📊 Class Probabilities")


            for animal, probability in zip(
                classes,
                probabilities
            ):

                percentage = probability * 100

                st.write(
                    f"{animal_emoji[animal]} "
                    f"**{animal.capitalize()}** "
                    f"— {percentage:.2f}%"
                )

                st.progress(
                    float(probability)
                )


        # =================================================
        # CONFIDENCE MESSAGE
        # =================================================

        st.markdown("<br>", unsafe_allow_html=True)

        if confidence >= 80:

            st.success(
                f"🎯 The model is highly confident that "
                f"this image belongs to the "
                f"**{display_name}** category."
            )

        elif confidence >= 60:

            st.warning(
                f"⚠️ The model predicts **{display_name}**, "
                f"but confidence is moderate. "
                f"Try uploading a clearer image."
            )

        else:

            st.warning(
                f"⚠️ The model predicts **{display_name}**, "
                f"but confidence is low. "
                f"The image may be ambiguous or different "
                f"from the training data."
            )


        # =================================================
        # HOW IT WORKS
        # =================================================

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(dedent("""
        <div class="info-box">

        <strong>ℹ️ How the prediction works</strong>

        <br><br>

        The uploaded image is converted to RGB format,
        resized to <strong>64 × 64 pixels</strong>,
        flattened into <strong>12,288 pixel features</strong>,
        and normalized before being passed to the
        Random Forest classifier.

        </div>
        """), unsafe_allow_html=True)


    except Exception as e:

        st.error(
            "Unable to process this image. "
            "Please try another image."
        )

        st.exception(e)


# =========================================================
# DEFAULT INFORMATION
# =========================================================

if uploaded_file is None:

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(dedent("""
    <div class="card">

    <div class="card-title">
    🐾 Supported Categories
    </div>

    <p>
    This model has been trained to recognize three
    categories of animals:
    </p>

    <br>

    🐱 <strong>Cat</strong> — Domestic cats

    <br><br>

    🐶 <strong>Dog</strong> — Domestic dogs

    <br><br>

    🦁 <strong>Wild</strong> — Wild animal images

    <br><br>

    <div class="info-box">

    For best results, upload a clear image with a
    single prominent animal and minimal background clutter.

    </div>

    </div>
    """), unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown(dedent("""
<div class="footer">

<strong>AI Animal Vision</strong>

<br>

Machine Learning Image Classification Project

<br>

Built with Python • Scikit-learn • Random Forest • Streamlit

</div>
"""), unsafe_allow_html=True)