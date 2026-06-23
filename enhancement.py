import cv2
import numpy as np
import os

def enhance_image(uploaded_file):
    # Convert Streamlit uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Invalid image. Please upload a valid file.")

    # CLAHE (contrast enhancement)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    # Denoising
    enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

    # Create output folder if not exists
    os.makedirs("outputs", exist_ok=True)

    output_path = "outputs/enhanced.jpg"
    cv2.imwrite(output_path, enhanced)

    return output_path
