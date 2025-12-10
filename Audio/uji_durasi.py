import os
import time
import json
import subprocess
from pathlib import Path

# === KONFIGURASI ===
audio_dir = Path("audio")       # Folder tempat file audio (WAV)
output_dir = Path("test_runtime")
os.makedirs(output_dir, exist_ok=True)

# === Fungsi: ambil durasi audio dengan ffprobe ===
def get_media_duration(path):
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", str(path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        info = json.loads(result.stdout)

        if "format" not in info or "duration" not in info["format"]:
            raise ValueError("Durasi tidak ditemukan di metadata")

        return float(info["format"]["duration"])
    except Exception as e:
        print(f"⚠️  Gagal baca durasi {path}: {e}")
        return None

# === Fungsi: proses audio (ganti dummy dengan pipeline STT kamu) ===
def proses_audio(path):
    start = time.time()

    # 🚀 Ganti ini ke pipeline kamu misal:
    # os.system(f'python run_pipeline.py --input "{path}"')
    # atau:
    # os.system(f'python model_whisper.py --input "{path}"')
    time.sleep(3)  # sementara dummy biar bisa test runtime cepat

    end = time.time()
    return end - start

# === Jalankan uji durasi semua audio ===
report = []
for file in os.listdir(audio_dir):
    if file.lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
        path = audio_dir / file
        print(f"▶️ Memproses {file} ...")

        durasi = get_media_duration(path)
        if not durasi:
            print(f"❌ Lewati {file}, durasi tidak bisa dibaca.")
            continue

        durasi_proses = proses_audio(path)
        rasio = durasi_proses / durasi

        status = "✅ OK" if rasio <= 2 else "⚠️ LAMBAT"
        report.append({
            "file": file,
            "durasi_media": round(durasi, 2),
            "durasi_proses": round(durasi_proses, 2),
            "rasio_proses_vs_media": round(rasio, 2),
            "status": status
        })

# === Tambahkan ringkasan ===
if report:
    rata_rasio = sum([r["rasio_proses_vs_media"] for r in report]) / len(report)
    semua_ok = all(r["status"] == "✅ OK" for r in report)
    report.append({
        "RINGKASAN": {
            "total_file_diuji": len(report),
            "rata_rata_rasio": round(rata_rasio, 2),
            "semua_ok": semua_ok
        }
    })

# === Simpan hasil laporan ===
out_path = output_dir / "runtime_report.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n🔥 Uji durasi selesai! Hasil disimpan di:", out_path)