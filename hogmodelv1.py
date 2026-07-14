

import os
import cv2
from ultralytics import YOLO


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_DIR, "input_images")
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "output_images")

MODEL_NAME = "yolov8n.pt"     
CONFIDENCE_THRESHOLD = 0.4   
PERSON_CLASS_ID = 0 # person         
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def load_model():

    print(f"Loading model '{MODEL_NAME}' (downloads automatically on first run)...")
    model = YOLO(MODEL_NAME)
    return model


def detect_people(model, image):

    results = model(image, verbose=False)[0]

    people = []
    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if class_id == PERSON_CLASS_ID and confidence >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            people.append((x1, y1, x2, y2, confidence))

    return people


def process_folder(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    model = load_model()

    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]

    if not image_files:
        print(f"No images found in '{input_folder}'.")
        return

    print(f"Found {len(image_files)} image(s). Starting detection...\n")

    for filename in image_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        image = cv2.imread(input_path)
        if image is None:
            print(f"[SKIPPED] Could not read: {filename}")
            continue

        detections = detect_people(model, image)

        for (x1, y1, x2, y2, confidence) in detections:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"person {confidence:.2f}"
            cv2.putText(image, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imwrite(output_path, image)
        print(f"[OK] {filename}: {len(detections)} person(s) detected -> saved to {output_path}")

    print("\nDone. All results saved to:", output_folder)


if __name__ == "__main__":
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)