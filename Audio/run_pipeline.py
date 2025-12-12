import os
import sys
import json
import argparse
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
import re
from collections import Counter
import ffmpeg
import whisper

# pyannote (diarization)
try:
    from pyannote.audio import Pipeline
except Exception:
    Pipeline = None

try:
    from boost_akurasi import clean_text, normalize_spoken_form, remove_repeated_words, soft_similarity
except Exception as e:
    print("Error importing boost_akurasi.py:", e)
    print("Make sure boost_akurasi.py is in the same folder.")
    def clean_text(t): return t.lower()
    def normalize_spoken_form(t): return t
    def remove_repeated_words(t): return t
    def soft_similarity(a,b): return 0.0

# Config
FILLER_WORDS = set([
    "um", "uh", "you know", "like", "basically", "actually", "so", "i mean",
    "right", "well", "okay", "hmm", "huh"
])

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def convert_webm_to_wav(input_video: str, output_wav: str):
    """Convert video.webm to mono 16k WAV"""
    try:
        (
            ffmpeg
            .input(input_video)
            .output(output_wav, ac=1, ar=16000, format='wav')
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        print("FFmpeg conversion failed:", getattr(e, 'stderr', str(e)))
        raise

# Whisper STT
def load_whisper_model(model_name="large-v3"):
    print(f"[{datetime.now().isoformat()}] Loading Whisper model: {model_name} (this may take a while)...")
    model = whisper.load_model(model_name)
    return model

def run_whisper_transcribe(model, wav_path: str, language="en"):
    """Run Whisper transcription and return raw text"""
    print(f"[{datetime.now().isoformat()}] Running Whisper STT on: {wav_path}")
    
    result = model.transcribe(str(wav_path), language=language, fp16=False)
    text = result.get("text", "").strip()
    return text

# Pyannote diarization
def load_diarization_pipeline():
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("HF_TOKEN not found in environment. Diarization will be skipped.")
        return None
    if Pipeline is None:
        print("pyannote.audio Pipeline not available. Skipping diarization.")
        return None
    try:
        print(f"[{datetime.now().isoformat()}] Loading pyannote diarization pipeline (pyannote/speaker-diarization@2.1)...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=hf_token)
        return pipeline

    except Exception as e:
        print("Failed to load pyannote pipeline:", str(e))
        return None

def run_diarization(pipeline, wav_path: str, min_speakers=1, max_speakers=3):
    """Return segments: [{'start':..,'end':..,'speaker':..}, ...]"""
    if pipeline is None:
        return []
    print(f"[{datetime.now().isoformat()}] Running diarization on: {wav_path}")
    diarization = pipeline(str(wav_path), min_speakers=min_speakers, max_speakers=max_speakers)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({"start": round(turn.start, 2), "end": round(turn.end, 2), "speaker": speaker})
    return segments

# Merge & Metrics
def sentences_from_transcript(text: str):
    text = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def merge_diarization_transcript(segments, transcript_text):
    """Simple sentence-wise rotation merging (consistent with your gabung_diarization)"""
    sentences = sentences_from_transcript(transcript_text)
    if not sentences:
        return []
    speakers = [s["speaker"] for s in segments] if segments else []
    if not speakers:
        speakers = ["SPEAKER_00"]
    merged = []
    for i, s in enumerate(sentences):
        speaker = speakers[i % len(speakers)]
        merged.append({"speaker": speaker, "text": s})
    return merged

def count_fillers_and_repetitions(raw_text: str):
    t = raw_text.lower()
    filler_count = 0
    for fw in FILLER_WORDS:
        filler_count += t.count(fw)
    tokens = re.findall(r"[a-z']+", t)
    repetition_count = 0
    prev = None
    for tok in tokens:
        if tok == prev:
            repetition_count += 1
        prev = tok
    return filler_count, repetition_count

def lexical_diversity(text: str):
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return 0.0, 0, 0
    total = len(tokens)
    unique = len(set(tokens))
    return unique / total, unique, total

def sentence_stats(sentences):
    if not sentences:
        return {"sentence_count": 0, "avg_sentence_len": 0.0}
    lens = [len(re.findall(r"[a-zA-Z0-9']+", s)) for s in sentences]
    return {"sentence_count": len(sentences), "avg_sentence_len": sum(lens) / len(lens)}

def pause_estimation_from_segments(segments, pause_threshold=0.5):
    if not segments:
        return 0
    segs = sorted(segments, key=lambda x: x["start"])
    pauses = 0
    for i in range(1, len(segs)):
        gap = segs[i]["start"] - segs[i-1]["end"]
        if gap > pause_threshold:
            pauses += 1
    return pauses

def speech_rate_wpm(word_count, duration_seconds):
    if not duration_seconds or duration_seconds <= 0:
        return 0.0
    minutes = duration_seconds / 60.0
    return word_count / minutes if minutes > 0 else 0.0

# durasi
def get_media_duration_seconds(path):
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(res.stdout.strip())
    except Exception:
        return None

# Main pipeline
def run_pipeline(video_path, out_dir="reports", temp_dir=None, whisper_model="large-v3", diar_min=1, diar_max=3):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp(prefix="pipeline_tmp_"))
    ensure_dir(temp_dir)

    transcripts_dir = Path("transcript")
    result_dir = Path("result")
    merged_dir = Path("merged")
    ensure_dir(transcripts_dir); ensure_dir(result_dir); ensure_dir(merged_dir)

    # 1) convert
    wav_path = temp_dir / (video_path.stem + ".wav")
    print(f"[{datetime.now().isoformat()}] Converting {video_path} -> {wav_path}")
    convert_webm_to_wav(str(video_path), str(wav_path))

    # 2) STT
    whisper_m = load_whisper_model(whisper_model)
    raw_transcript = run_whisper_transcribe(whisper_m, str(wav_path), language="en")

    raw_transcript_path = transcripts_dir / (video_path.stem + "_raw.txt")
    raw_transcript_path.write_text(raw_transcript, encoding="utf-8")

    # 3) diarization (optional)
    pipeline = load_diarization_pipeline()
    try:
        segments = run_diarization(pipeline, str(wav_path), min_speakers=diar_min, max_speakers=diar_max)
    except Exception as e:
        print("Diarization runtime error (continuing without diarization):", e)
        segments = []

    diar_out = result_dir / (video_path.stem + "_diarization.json")
    diar_out.write_text(json.dumps(segments, indent=2), encoding="utf-8")

    # 4) merge diarization + transcript (simple)
    merged = merge_diarization_transcript(segments, raw_transcript)
    merged_out = merged_dir / (video_path.stem + "_merged.json")
    merged_out.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    # 5) NLP metrics
    norm_text = normalize_spoken_form(raw_transcript)
    clean = clean_text(norm_text)
    dedup_text = remove_repeated_words(clean)

    tokens = re.findall(r"[a-zA-Z0-9']+", clean)
    word_count = len(tokens)

    sentences = sentences_from_transcript(raw_transcript)
    sent_stats = sentence_stats(sentences)

    diversity, unique_words, total_tokens = lexical_diversity(clean)
    filler_count, repetition_count = count_fillers_and_repetitions(raw_transcript)

    duration_seconds = get_media_duration_seconds(wav_path)
    wpm = speech_rate_wpm(word_count, duration_seconds if duration_seconds else 1.0)
    pause_count = pause_estimation_from_segments(segments, pause_threshold=0.5)

    nlp_metrics = {
        "word_count": word_count,
        "unique_words": unique_words,
        "lexical_diversity": round(diversity, 3),
        "speech_rate_wpm": round(wpm, 2),
        "sentence_count": sent_stats["sentence_count"],
        "average_sentence_length": round(sent_stats["avg_sentence_len"], 2),
        "filler_words_count": filler_count,
        "repetition_count": repetition_count,
        "pause_estimation_count": pause_count
    }

    audio_metrics = {
        "duration_seconds": round(duration_seconds, 2) if duration_seconds else None,
        "audio_path": str(wav_path)
    }

    report = {
        "video": str(video_path),
        "video_stem": video_path.stem,
        "speakers_detected": len(set([s["speaker"] for s in segments])) if segments else 1,
        "segments": segments,
        "transcript": raw_transcript,
        "merged_transcript": merged,
        "nlp_metrics": nlp_metrics,
        "audio_metrics": audio_metrics,
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }

    out_path = out_dir / (video_path.stem + "_report.json")
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{datetime.now().isoformat()}] Report saved -> {out_path}")

    return out_path

# -------------------------
# CLI (supports --video & --folder)
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Run end-to-end interview analysis pipeline.")
    parser.add_argument("--video", "-v", help="Path to single .webm video")
    parser.add_argument("--folder", "-f", help="Folder containing .webm videos")
    parser.add_argument("--out_dir", "-o", default="reports", help="Output reports folder")
    parser.add_argument("--whisper_model", default="large-v3", help="Whisper model name (default: large-v3)")
    parser.add_argument("--diar_min", type=int, default=1, help="Diarization min speakers")
    parser.add_argument("--diar_max", type=int, default=3, help="Diarization max speakers")
    args = parser.parse_args()

    if args.video:
        print("▶️ Processing single video:", args.video)
        run_pipeline(args.video, out_dir=args.out_dir, whisper_model=args.whisper_model, diar_min=args.diar_min, diar_max=args.diar_max)
        return

    if args.folder:
        folder_path = Path(args.folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        video_files = sorted(list(folder_path.glob("*.webm")))
        if not video_files:
            print("⚠️ No .webm files found in folder.")
            return
        print(f"📁 Batch mode: found {len(video_files)} .webm files.")
        for vid in video_files:
            print("\n==============================")
            print(f"▶️ Processing: {vid.name}")
            print("==============================")
            try:
                run_pipeline(vid, out_dir=args.out_dir, whisper_model=args.whisper_model, diar_min=args.diar_min, diar_max=args.diar_max)
            except Exception as e:
                print(f"❌ Failed to process {vid.name}: {e}")
        print("\n🔥 All files in folder processed.")
        return

    # no input provided
    print("⚠️ No input supplied. Use one of:")
    print("   python run_pipeline.py --video path/to/file.webm")
    print("   python run_pipeline.py --folder path/to/folder")

if __name__ == "__main__":
    main()