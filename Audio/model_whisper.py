import os
from pathlib import Path
import whisper


AUDIO_DIR = "audio"
TRANSCRIPT_DIR = "transcript"

Path(TRANSCRIPT_DIR).mkdir(exist_ok=True)

print("Loading Whisper model...")
model = whisper.load_model("large-v3")
print("Model loaded.")

for file in os.listdir(AUDIO_DIR):
    if file.lower().endswith(".wav"):
        audio_path = os.path.join(AUDIO_DIR, file)
        print("Transcribing:", audio_path)
        
        result = model.transcribe(
            audio_path,
            language="en",
            fp16=False,
            beam_size=5,
            temperature=0.0
)
        
        out_path = os.path.join(TRANSCRIPT_DIR, file.replace(".wav", "_raw.txt"))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        
        print("Saved:", out_path)