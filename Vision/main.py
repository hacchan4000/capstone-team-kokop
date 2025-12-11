import os
import tempfile

import cv2
import numpy as np

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from keras.models import load_model

model_path = "/Users/mac/Desktop/CAPSTONE/capstone-team-kokop/EyeTrackerModel.h5"
model = load_model(model_path)

app = FastAPI(title="Iris_Tracking_API")

def penentuArah(x1,y1,x2,y2): #fungsi untuk nulis arah pandang user
    x_center = (x2+x1)/2
    y_center = (y2+y1)/2
    
    if x_center < 0.35:   
        sumbuX = "Kiri"
    elif x_center > 0.65:
        sumbuX = "Kanan"
    else:
        sumbuX = "Tengah"
        
    if y_center < 0.35:   
        sumbuY = "Atas"
    elif y_center > 0.65:
        sumbuY = "Bawah"
    else:
        sumbuY = "Tengah"
        
    if sumbuX == "Tengah" and sumbuY == "Tengah":
        return "Tengah"
    else:
        return f"{sumbuY}-{sumbuX}"
    

@app.post('predict-video')
async def predict_video(file: UploadFile = File(...)):
    # save ke file sementara
    dir_sementara = tempfile.mkdtemp()
    path_video = os.path.join(dir_sementara, file.filename)
    with open(path_video, 'wb') as f:
        f.write(await file.read())
    
    #buka video
    video = cv2.VideoCapture(path_video)
    if not video.isOpened():
        return JSONResponse(
            {"error":"gabisa buka videonya wle"},
            status_code=400
        )
    
    
    prediksi_model = [] #list tempat nyimpen semua prediksi model
    
    # jalaninin loop model
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        #crop
        crop_frame = frame[50:500, 50:500]
        #process
        gambar = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
        gambar = cv2.resize(gambar,(250,250))
        gambar = np.expand_dims(gambar/255.0, axis=0)
        #predict
        prediksi = model.predict(gambar)
        x1,y1,x2,y2 = prediksi[0]
        
        arah = penentuArah(x1,y1,x2,y2)
        prediksi_model.append(arah)
    video.release()
        
    hasilAkhir = max(set(prediksi_model), key=prediksi_model.count)
    return {
            "Hasil akhir" : hasilAkhir
            }
    
    pass

@app.get("/")
def home():
    return {"message": "Iris Eye Tracker API Running 👍"}

#run server

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)


