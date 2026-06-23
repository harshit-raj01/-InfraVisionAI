import cv2

def colorize_image(image_path):

    img = cv2.imread(image_path)

    colored = cv2.applyColorMap(
        img,
        cv2.COLORMAP_TURBO
    )

    output = "outputs/colorized.jpg"

    cv2.imwrite(output, colored)

    return output