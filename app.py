import streamlit as st

from enhancement import enhance_image
from colorization import colorize_image
from detection import detect_objects

st.set_page_config(page_title="InfraVision AI")

st.title("InfraVision AI")
st.subheader("Infrared Image Enhancement and Object Interpretation")

uploaded = st.file_uploader(
    "Upload Infrared Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:

    path = "uploads/input.jpg"

    with open(path, "wb") as f:
        f.write(uploaded.read())

    st.subheader("Original Image")
    st.image(path)

    enhanced = enhance_image(path)

    st.subheader("Enhanced Image")
    st.image(enhanced)

    colorized = colorize_image(enhanced)

    st.subheader("Colorized Image")
    st.image(colorized)

    detected, labels = detect_objects(colorized)

    st.subheader("Detected Objects")
    st.image(detected)

    st.write(labels)

    st.subheader("AI Interpretation")

    summary = (
        f"The scene contains: "
        f"{', '.join(labels)}"
    )

    st.success(summary)