import streamlit as st

from enhancement import enhance_image
from colorization import colorize_image
from detection import detect_objects

st.set_page_config(page_title="InfraVision AI")

st.title("InfraVision AI")
st.subheader("Infrared Image Enhancement and Object Interpretation")

uploaded = st.file_uploader("Upload Infrared Image", type=["jpg", "jpeg", "png"])

if uploaded is not None:

    st.subheader("Original Image")
    st.image(uploaded)

    # STEP 1: Enhancement
    enhanced = enhance_image(uploaded)
    st.subheader("Enhanced Image")
    st.image(enhanced)

    # STEP 2: Colorization
    colorized = colorize_image(enhanced)
    st.subheader("Colorized Image")
    st.image(colorized)

    # STEP 3: Detection
    detected, labels = detect_objects(colorized)
    st.subheader("Detected Objects")
    st.image(detected)

    st.write(labels)

    # STEP 4: Summary
    summary = "The scene contains: " + ", ".join(labels)
    st.subheader("AI Interpretation")
    st.success(summary)
