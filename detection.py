from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_objects(image_path):

    results = model(image_path)

    output = "outputs/detected.jpg"

    results[0].save(filename=output)

    labels = []

    for box in results[0].boxes:
        cls = int(box.cls)
        labels.append(model.names[cls])

    return output, list(set(labels))