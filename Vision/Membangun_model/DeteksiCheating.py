import os
import mlflow
import cv2
import numpy as np

model = 'EyeTrackingModel.h5'

def load_model(model_path):
    pass
       
model = load_model('') 

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _ , frame = cap.read()
    
    frame = frame[50:500,50:500,:] 
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_img, (250,250))
    
    yhat = model.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat[0,:4]
    
    cv2.circle(frame, tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)), 2, (255,0,0), -1)
    cv2.circle(frame, tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), 2, (0,255,0), -1)
    
    cv2.imshow('EyeTrack', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

ratio = 100

myModel = 'EyeTrackerModel.h5'

def arah():
    direction = ''
    if ratio <= 0.35 :
        direction = "kiri"
    elif 0.35 <= ratio and ratio <= 0.65:
        direction = "tengah"
    else:
        direction="kanan"
    return direction
    
def start_clock():
    pass