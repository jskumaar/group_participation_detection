import subprocess
import os

# === CONFIG ===
# Update paths as needed
ORIGINAL_VIDEO = r"C:\Users\sures\Desktop\children_pilot_2_360.mp4"
SHIFTED_VIDEO = r"C:\Users\sures\Desktop\children_pilot_2_360_corrected.mp4"
OUTPUT_VIDEO  = r"C:\Users\sures\Desktop\children_pilot_2_360_corrected_with_audio.mp4"

# === CHECK INPUT FILES ===
for path in [ORIGINAL_VIDEO, SHIFTED_VIDEO]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

# === MERGE VIDEO (SHIFTED) + AUDIO (ORIGINAL) ===
cmd = [
    "ffmpeg",
    "-y",  # overwrite without asking
    "-i", SHIFTED_VIDEO,  # shifted video (no audio)
    "-i", ORIGINAL_VIDEO,  # original video (has audio)
    "-c:v", "copy",  # copy video stream (no re-encoding)
    "-map", "0:v:0",  # take video from first input
    "-map", "1:a:0",  # take audio from second input
    "-shortest",      # stop at the shortest stream
    OUTPUT_VIDEO
]

print("Merging audio from original video...")
subprocess.run(cmd, check=True)
print(f"✅ Done! Saved merged video → {OUTPUT_VIDEO}")
