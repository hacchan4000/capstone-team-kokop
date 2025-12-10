import json, os

diar_dir = "result"
trans_dir = "transcript"
output_dir = "merged"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(diar_dir):
    if filename.endswith("_diarization.json"):
        base = filename.replace("_diarization.json", "")
        diar_path = os.path.join(diar_dir, filename)
        trans_path = os.path.join(trans_dir, base + "_raw.txt")
        out_path = os.path.join(output_dir, base + "_merged.json")

        if not os.path.exists(trans_path):
            print(f"⚠️ Transkrip untuk {base} tidak ditemukan, lewati.")
            continue

        # Baca file diarization & teks mentahan
        diar = json.load(open(diar_path, "r", encoding="utf-8"))
        with open(trans_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Bagi teks jadi kalimat-kalimat biar bisa disisipin speaker
        sentences = [t.strip() for t in text.replace("\n", " ").split(". ") if t.strip()]

        # Ambil pembicara dari diarization (gilir-gantian aja)
        merged = []
        speakers = [d["speaker"] for d in diar]
        if not speakers:
            speakers = ["SPEAKER_00"]
        for i, s in enumerate(sentences):
            speaker = speakers[i % len(speakers)]
            merged.append({"speaker": speaker, "text": s})

        json.dump(merged, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"✅ Gabung selesai: {out_path}")

print("🔥 Semua file sudah digabung dengan versi simpel!")