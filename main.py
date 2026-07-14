

import os
import cv2
import imutils
from imutils.object_detection import non_max_suppression
import numpy as np



INPUT_FOLDER = "C:\\Users\\produ\\Ganpat\\Learning\\Python\\d1\\src\\input_images"
OUTPUT_FOLDER = "output_images"
RESIZE_WIDTH = 400          
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def get_hog_detector():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def detect_people(hog, image):
    (rects, weights) = hog.detectMultiScale(
        image,
        winStride=(4, 4), 
        padding=(8, 8), 
        scale=1.05
    )

    # for NMS
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

    picked = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    return picked


def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    hog = get_hog_detector()

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

        image = imutils.resize(image, width=min(RESIZE_WIDTH, image.shape[1]))

        boxes = detect_people(hog, image)

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imwrite(output_path, image)
        print(f"[OK] {filename}: {len(boxes)} person(es) detected -> saved to {output_path}")

    print("\nDone. All results saved to:", output_folder)


if __name__ == "__main__":
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)