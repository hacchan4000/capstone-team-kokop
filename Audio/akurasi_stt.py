# akurasi_stt.py

from jiwer import wer, cer
from boost_akurasi import (
    clean_text,
    normalize_spoken_form,
    remove_repeated_words,
    soft_similarity
)

GROUND_TRUTH = "groundtruth/interview5_gt.txt"
STT_RESULT   = "transcript/interview_question_5_raw.txt"

with open(GROUND_TRUTH, "r", encoding="utf-8") as f:
    gt_raw = f.read()

with open(STT_RESULT, "r", encoding="utf-8") as f:
    pred_raw = f.read()

def prepare(text):
    text = normalize_spoken_form(text)
    text = clean_text(text)
    text = remove_repeated_words(text)
    return text

gt = prepare(gt_raw)
pred = prepare(pred_raw)

# --- Metrics ---
wer_value = wer(gt, pred)
cer_value = cer(gt, pred)

wer_acc = max(0, (1 - wer_value) * 100)
cer_acc = max(0, (1 - cer_value) * 100)
soft_sim = soft_similarity(gt_raw, pred_raw)

composite = (wer_acc * 0.6) + (cer_acc * 0.2) + (soft_sim * 0.2)

# --- Report ---
print("\n=== STT ACCURACY REPORT (REPORT-READY) ===")
print(f"GT Words        : {len(gt.split())}")
print(f"Pred Words      : {len(pred.split())}")
print(f"WER Value       : {wer_value:.3f}")
print(f"WER Accuracy    : {wer_acc:.2f}%")
print(f"CER Accuracy    : {cer_acc:.2f}%")
print(f"Soft Similarity : {soft_sim:.2f}%")
print(f"Composite Score : {composite:.2f}%")