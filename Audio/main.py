import os
import json
import re
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

import whisper
import ffmpeg
from fastapi import FastAPI, UploadFile, File, HTTPException

# --------------------
# App
# --------------------
app = FastAPI(title="Audio_Analyzer_API")

# --------------------
# Load Whisper ONCE
# --------------------

WHISPER_MODEL = None

@app.on_event("startup")
def load_models():
    global WHISPER_MODEL
    print("🔊 Loading Whisper medium...")
    WHISPER_MODEL = whisper.load_model("medium")
    print("✅ Whisper medium loaded")
# --------------------
# Optional pyannote
# --------------------
try:
    from pyannote.audio import Pipeline
except Exception:
    Pipeline = None

def load_diarization_pipeline():
    token = os.getenv("HF_TOKEN")
    if not token or Pipeline is None:
        return None
    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1",
        use_auth_token=token
    )

DIAR_PIPELINE = load_diarization_pipeline()

# --------------------
# Utils
# --------------------
FILLER_WORDS = {
    "um", "uh", "you know", "like", "basically", "actually",
    "so", "i mean", "right", "well", "okay", "hmm", "huh"
}

def convert_to_wav(video_path: str, wav_path: str):
    ffmpeg.input(video_path).output(
        wav_path, ac=1, ar=16000, format="wav"
    ).overwrite_output().run(quiet=True)

def sentences_from_transcript(text: str):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def lexical_diversity(text: str):
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return 0.0, 0, 0
    return len(set(tokens)) / len(tokens), len(set(tokens)), len(tokens)

def count_fillers(text: str):
    t = text.lower()
    return sum(t.count(w) for w in FILLER_WORDS)

def get_duration(path: str):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return float(r.stdout.strip())

# --------------------
# API Endpoint
# --------------------
@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    if not file.filename.endswith((".webm", ".mp4", ".wav")):
        raise HTTPException(status_code=400, detail="Invalid file type")

    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / file.filename
        wav_path = Path(tmp) / "audio.wav"

        video_path.write_bytes(await file.read())

        # Convert
        convert_to_wav(str(video_path), str(wav_path))

        # Whisper
        result = WHISPER_MODEL.transcribe(str(wav_path), fp16=False)
        transcript = result["text"].strip()

        # Diarization (optional)
        segments = []
        if DIAR_PIPELINE:
            diar = DIAR_PIPELINE(str(wav_path))
            for turn, _, speaker in diar.itertracks(yield_label=True):
                segments.append({
                    "speaker": speaker,
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2),
                })

        # Metrics
        sentences = sentences_from_transcript(transcript)
        diversity, uniq, total = lexical_diversity(transcript)
        duration = get_duration(str(wav_path))

        return {
            "transcript": transcript,
            "sentences": sentences,
            "nlp_metrics": {
                "word_count": total,
                "unique_words": uniq,
                "lexical_diversity": round(diversity, 3),
                "filler_words": count_fillers(transcript),
                "speech_rate_wpm": round((total / duration) * 60, 2),
            },
            "audio_metrics": {
                "duration_seconds": round(duration, 2),
            },
            "diarization": segments,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

# --------------------
# Run
# --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002)
