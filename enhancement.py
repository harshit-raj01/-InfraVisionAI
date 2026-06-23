import cv2

def enhance_image(image_path):
    img = cv2.imread(image_path, 0)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(img)

    enhanced = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        10,
        7,
        21
    )

    output = "outputs/enhanced.jpg"

    cv2.imwrite(output, enhanced)

    return output