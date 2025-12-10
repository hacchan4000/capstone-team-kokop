import cv2
import numpy as np
from keras.models import load_model
import pandas as pd
import time
import streamlit as st

# Load model
model_path = '/Users/mac/Desktop/CAPSTONE/capstone-team-kokop/EyeTrackerModel.h5'
model = load_model(model_path)

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def arah_full(x1, y1, x2, y2):
    """
    Detect 5-way gaze direction:
    centre, top-left, top-right, down-left, down-right
    """

    # Use midpoint of the two iris points
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    # Horizontal
    if x_center < 0.35:
        h = "left"
    elif x_center > 0.65:
        h = "right"
    else:
        h = "center"

    # Vertical
    if y_center < 0.35:
        v = "top"
    elif y_center > 0.65:
        v = "down"
    else:
        v = "center"

    # Combine
    if h == "center" and v == "center":
        return "center"
    else:
        return f"{v}-{h}"


# ---------------------------------------------------------
# Collect gaze for 10 seconds
# ---------------------------------------------------------

hasilAkhir = []

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("❌ Could not open webcam")

end_time = time.time() + 30

while time.time() < end_time:

    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    # Crop ROI
    frame_crop = frame[50:500, 50:500]

    # Preprocess
    rgb_img = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_img, (250, 250))
    input_img = np.expand_dims(resized / 255.0, axis=0)

    # Predict
    yhat = model.predict(input_img)
    x1, y1, x2, y2 = yhat[0, :4]

    # Get direction
    direction_text = arah_full(x1, y1, x2, y2)
    hasilAkhir.append(direction_text)

    # Draw points
    p1 = tuple((np.array([x1, y1]) * 450).astype(int))
    p2 = tuple((np.array([x2, y2]) * 450).astype(int))

    cv2.circle(frame_crop, p1, 5, (255, 0, 0), -1)
    cv2.circle(frame_crop, p2, 5, (0, 255, 0), -1)

    # Draw direction text
    cv2.putText(frame_crop, direction_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow('Eye Tracking', frame_crop)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# ---------------------------------------------------------
# Output analysis
# ---------------------------------------------------------

df = pd.DataFrame(hasilAkhir, columns=["direction"])
print("\n📊 Frekuensi Arah:\n")
print(df.value_counts())


