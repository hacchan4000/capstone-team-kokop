# AI-Powered Interview Assessment System (MVP)

This project is a working prototype for an AI-powered interview assessment system.
It processes interview videos/audio and automatically generates evaluation reports.

## Features
- English Speech-to-Text (Whisper Large v3)
- Speaker diarization (Pyannote)
- STT accuracy evaluation (>90%)
- Automatic interview evaluation report (NLP-based)
- Runtime efficiency testing (≤ 2× media duration)

## Project Pipeline
1. Convert interview video (.webm) to audio (.wav)
2. Run Speech-to-Text (Whisper)
3. Boost STT accuracy (text normalization)
4. Speaker diarization
5. Merge diarization and transcript
6. Automatic evaluation report generation

## Use Case Success Indicators
- STT accuracy > 90%
- 100% of provided samples processed
- Runtime ≤ 2× interview duration