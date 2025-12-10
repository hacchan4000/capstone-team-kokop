from pyannote.audio import Pipeline
import json, os

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# 2. Load pipeline diarization versi stabil (lebih cocok untuk MVP)
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HUGGINGFACE_TOKEN)

# 3. Path input/output
input_dir = "audio"
output_dir = "result"
os.makedirs(output_dir, exist_ok=True)

# 4. Loop semua file .wav
for filename in os.listdir(input_dir):
    if filename.endswith(".wav"):
        audio_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".wav", "_diarization.json"))
        print(f"🎧 Memproses {filename}...")

        # Jalankan diarization (batasi ke 2–3 pembicara agar hasilnya realistis)
        diarization = pipeline(audio_path, min_speakers=2, max_speakers=3)

        # Simpan hasil ke JSON
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker
            })

        with open(output_path, "w") as f:
            json.dump(segments, f, indent=2)

        print(f"✅ Selesai: {output_path}")

print("🔥 Semua file sudah diproses!")