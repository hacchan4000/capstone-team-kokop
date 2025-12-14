import json
import os
import tempfile
import shutil
import smtplib
from email.message import EmailMessage
import cv2

import numpy as np
from pytz import utc
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
import httpx
import asyncio

from dotenv import load_dotenv
load_dotenv()

# ---- CONFIG ----
MODEL_PATH = "/Users/mac/Desktop/CAPSTONE/capstone-team-kokop/EyeTrackerModel.h5"
MODEL = load_model(MODEL_PATH)

app = FastAPI(title="Iris_Tracking_API")

# simple CORS so your Next.js frontend (http://localhost:3000) can call this

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # atau origin React kamu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def penentuArah(x1, y1, x2, y2):
    x_center = (x2 + x1) / 2
    y_center = (y2 + y1) / 2

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

def send_email_smtp(subject: str, body: str, to_emails: list[str]):
    """
    Simple SMTP sender. Uses env vars:
      EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_FROM
    """
    host = os.environ.get("EMAIL_HOST")
    port = int(os.environ.get("EMAIL_PORT", "587"))
    user = os.environ.get("EMAIL_USER")
    password = os.environ.get("EMAIL_PASS")
    sender = os.environ.get("EMAIL_FROM", user)

    if not all([host, user, password]):
        # missing config; don't raise but log (in production use proper logging)
        print("Email config missing; cannot send email.")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(to_emails)
    msg.set_content(body)

    with smtplib.SMTP(host, port) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)
        print("Email sent to:", to_emails)
 
async def run_vision_model(video_path: str):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise RuntimeError("Tidak bisa membuka file video")

    prediksi_model = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        h, w, _ = frame.shape
        crop = frame[50:min(300, h), 50:min(300, w)]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        img = cv2.resize(crop, (250, 250))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        pred = MODEL.predict(img, verbose=0)
        x1, y1, x2, y2 = pred[0]
        prediksi_model.append(penentuArah(x1, y1, x2, y2))

    video.release()

    if not prediksi_model:
        raise RuntimeError("Tidak ada frame valid")

    hasil_akhir = max(set(prediksi_model), key=prediksi_model.count)
    counts = {k: prediksi_model.count(k) for k in set(prediksi_model)}

    return {
        "result": hasil_akhir,
        "raw_counts": counts
    }


async def run_audio_model(path_video: str):
    url = "http://localhost:8002/predict-audio"

    timeout = httpx.Timeout(
        connect=10.0,
        read=600.0,   # allow long inference
        write=10.0,
        pool=10.0,
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            with open(path_video, "rb") as f:
                files = {"file": ("video.mp4", f, "video/mp4")}
                r = await client.post(url, files=files)

            r.raise_for_status()
            return r.json()

        except httpx.ReadTimeout:
            return {"error": "Audio model timeout"}

        except httpx.HTTPError as e:
            return {"error": str(e)}

# ---- Endpoint that accepts form fields + file ----
@app.post("/get_results")

async def get_results(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    email: str = Form(...),
    email2: str = Form(...),
    file: UploadFile = File(...)
):
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, file.filename)

    try:
        # save file ONCE
        with open(video_path, "wb") as f:
            f.write(await file.read())

        # run BOTH models
        
        vision_task = asyncio.create_task(run_vision_model(video_path))
        audio_task = asyncio.create_task(run_audio_model(video_path))

        vision_result = await vision_task
        audio_result = await audio_task


        combined_result = {
            "vision": vision_result,
            "audio": audio_result
        }


        # email content
        subject = f"AI Interview Analysis Result – {name}"

        body = f"""
        Hello {name},

        Here are the combined AI analysis results:

        --- Vision Model ---
        Result: {vision_result["result"]}
        Counts: {json.dumps(vision_result["raw_counts"], indent=2)}

        --- Audio Model ---
        Result: {json.dumps(audio_result, indent=2)}

        Regards,
        SmartHire AI System
        """



        recipients = [email2, email]
        background_tasks.add_task(
            send_email_smtp,
            subject,
            body,
            recipients
        )

        return {
            "status": "ok",
            "name": name,
            "email": email,
            "recruiter_email": email2,
            "combined_result": combined_result,
            "message": "Vision & Audio prediction done. Email sent."
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)



@app.get("/")
def home():
    return {"message": "Iris Eye Tracker API Running 👍"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
