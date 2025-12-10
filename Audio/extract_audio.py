import os
import ffmpeg
from pathlib import Path

VIDEOS_DIR = "video"   # sesuai folder video kamu
OUTPUT_DIR = "audio"

Path(OUTPUT_DIR).mkdir(exist_ok=True)

def convert(video_path, out_path):
    (
        ffmpeg
        .input(video_path)
        .output(out_path, ac=1, ar=16000, format='wav')
        .overwrite_output()
        .run()
    )
    print(f"Converted: {video_path} -> {out_path}")

if __name__ == "__main__":
    for file in os.listdir(VIDEOS_DIR):
        if file.endswith(".webm"):
            input_path = os.path.join(VIDEOS_DIR, file)
            output_path = os.path.join(OUTPUT_DIR, file.replace(".webm", ".wav"))
            convert(input_path, output_path)
