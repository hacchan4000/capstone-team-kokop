import os
import tempfile
import shutil
import smtplib
from email.message import EmailMessage
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model

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

# ---- Endpoint that accepts form fields + file ----
@app.post("/predict-video")
async def predict_video(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    email: str = Form(...),
    email2: str = Form(...),
    file: UploadFile = File(...)
):
    dir_sementara = tempfile.mkdtemp()
    path_video = os.path.join(dir_sementara, file.filename)
    try:
        with open(path_video, "wb") as f:
            f.write(await file.read())

        video = cv2.VideoCapture(path_video)
        if not video.isOpened():
            return JSONResponse({"error": "Tidak bisa membuka file video"}, status_code=400)

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
            img = np.expand_dims(img, axis=0)  # shape: (1,250,250,3)

            pred = MODEL.predict(img, verbose=0)
            x1, y1, x2, y2 = pred[0]
            prediksi_model.append(penentuArah(x1, y1, x2, y2))


        video.release()

        if not prediksi_model:
            return JSONResponse({"error": "Video tidak mengandung frame yang bisa diproses."}, status_code=400)

        hasilAkhir = max(set(prediksi_model), key=prediksi_model.count)
        counts = {k: prediksi_model.count(k) for k in set(prediksi_model)}

        # prepare email content
        subject = f"Iris Tracking Result for {name}"
        body = f"Hello,\n\nThe model prediction for {name} is: {hasilAkhir}\n\nCounts: {counts}\n\nRegards,\nIris Tracker API"

        # send to recruiter and user in background (so response is immediate)
        recipients = [email2]
        # optional: cc the user
        if email:
            recipients.append(email)

        background_tasks.add_task(send_email_smtp, subject, body, recipients)

        return {
            "status": "ok",
            "name": name,
            "email": email,
            "recruiter_email": email2,
            "result": hasilAkhir,
            "raw_counts": counts,
            "message": "Prediction done. Email scheduled."
        }

    finally:
        try:
            shutil.rmtree(dir_sementara)
        except Exception:
            pass

@app.get("/")
def home():
    return {"message": "Iris Eye Tracker API Running 👍"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
