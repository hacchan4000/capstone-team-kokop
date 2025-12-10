import cv2
import numpy as np
from keras.models import load_model

# Load trained model
path_model = '/Users/mac/Desktop/CAPSTONE/capstone-team-kokop/EyeTrackerModel.h5'
model = load_model(path_model)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("❌ Could not open webcam")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Failed to capture frame")
        continue

    # Crop ROI (450x450)
    frame_crop = frame[50:500, 50:500]

    # Preprocess
    rgb_img = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_img, (250, 250))
    input_img = np.expand_dims(resized / 255.0, axis=0)

    # Predict keypoints
    yhat = model.predict(input_img)
    sample_coords = yhat[0, :4]   # x1,y1,x2,y2 normalized (0–1)

    # Map back to 450x450 crop
    p1 = tuple((sample_coords[:2] * 450).astype(int))
    p2 = tuple((sample_coords[2:] * 450).astype(int))

    # Draw
    cv2.circle(frame_crop, p1, 4, (255, 0, 0), -1)  # first point
    cv2.circle(frame_crop, p2, 4, (0, 255, 0), -1)  # second point

    cv2.imshow('EyeTrack', frame_crop)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
