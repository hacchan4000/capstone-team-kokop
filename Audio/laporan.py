# ============================================================
# laporan.py
# Automatic Interview Evaluation Report (MVP Version)
# ============================================================

import json
import os
import re
from collections import Counter

# ===============================
# KONFIGURASI
# ===============================
MERGED_DIR = "merged"
OUTPUT_DIR = "report"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILLER_WORDS = {
    "uh", "um", "yeah", "like", "you know", "actually"
}

# ===============================
# NLP FEATURE EXTRACTION
# ===============================
def extract_features(data):
    full_text = " ".join([d["text"] for d in data]).lower()
    words = re.findall(r"\b[a-z]+\b", full_text)

    total_words = len(words)
    unique_words = len(set(words))
    filler_count = sum(1 for w in words if w in FILLER_WORDS)

    speaker_counts = Counter(d["speaker"] for d in data)
    speaker_total = len(speaker_counts)

    return {
        "total_words": total_words,
        "unique_words": unique_words,
        "lexical_richness": round(unique_words / max(total_words, 1), 3),
        "filler_ratio": round(filler_count / max(total_words, 1), 3),
        "speaker_count": speaker_total,
        "speaker_distribution": dict(speaker_counts)
    }

# ===============================
# RULE-BASED ASSESSMENT SCORING
# ===============================
def calculate_scores(features):
    fluency = max(0, 100 - features["filler_ratio"] * 1200)
    clarity = min(100, features["lexical_richness"] * 300)
    confidence = 100 if features["speaker_count"] == 1 else 85

    overall = round((fluency + clarity + confidence) / 3, 1)

    return {
        "fluency_score": round(fluency, 1),
        "clarity_score": round(clarity, 1),
        "confidence_score": confidence,
        "overall_score": overall
    }

# ===============================
# MAIN PIPELINE
# ===============================
def generate_report(merged_path):
    filename = os.path.basename(merged_path).replace("_merged.json", "")
    data = json.load(open(merged_path, "r", encoding="utf-8"))

    features = extract_features(data)
    scores = calculate_scores(features)

    report = {
        "file": filename,
        "features": features,
        "assessment": scores
    }

    out_path = os.path.join(OUTPUT_DIR, f"{filename}_report.json")
    json.dump(report, open(out_path, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print(f"✅ Laporan evaluasi otomatis dibuat: {out_path}")

# ===============================
# BATCH PROCESS SEMUA FILE
# ===============================
if __name__ == "__main__":
    for file in os.listdir(MERGED_DIR):
        if file.endswith("_merged.json"):
            generate_report(os.path.join(MERGED_DIR, file))

    print("\n🎯 SEMUA LAPORAN BERHASIL DIBUAT")