import cv2
import os

def enhance_image(image_path):

    # Read image safely
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image could not be loaded. Check file path or upload.")

    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    # Denoising
    enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

    # Ensure output folder exists
    os.makedirs("outputs", exist_ok=True)

    output_path = os.path.join("outputs", "enhanced.jpg")

    cv2.imwrite(output_path, enhanced)

    return output_path
