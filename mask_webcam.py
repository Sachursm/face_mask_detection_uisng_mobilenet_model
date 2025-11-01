import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "mask_detector_model.keras"   # <- your saved model path
CLASS_NAMES = ["with_mask", "without_mask"]  # match your training order
IMG_SIZE = (224, 224)

# Load the trained model
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded successfully!")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_and_predict_mask(frame):
    """
    Detect faces in the frame, crop each, and predict mask or no mask.
    Returns the processed frame with labels drawn.
    """
    # Convert frame to grayscale (for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # Crop and preprocess the face
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, IMG_SIZE)
        face_array = np.expand_dims(face_resized, axis=0)
        face_array = preprocess_input(face_array)

        # Predict mask / no mask
        preds = model.predict(face_array, verbose=0)
        label_index = np.argmax(preds)
        label = CLASS_NAMES[label_index]
        confidence = preds[0][label_index]

        # Choose color: green = with mask, red = without mask
        color = (0, 255, 0) if label == "with_mask" else (0, 0, 255)

        # Draw rectangle and label
        cv2.putText(frame, f"{label}: {confidence*100:.2f}%", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    return frame


# -----------------------------
# MAIN LOOP (Webcam)
# -----------------------------
print("[INFO] Starting webcam...")
cap = cv2.VideoCapture(0)  # change to 1 if you have multiple cameras
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    # Detect + predict
    output = detect_and_predict_mask(frame)

    # Show the result
    cv2.imshow("Mask Detection", output)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Webcam closed.")
