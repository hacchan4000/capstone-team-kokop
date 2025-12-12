# model whisper di serve menggunakann FAST API

import whisper
import tempfile
import uvicorn


from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="Audio_analyzer")
myModel = whisper.load_model("large-v3")

@app.post("/analyze-audio")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    result = myModel.transcribe(tmp_path, fp16=False)
    return {"text": result["text"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
