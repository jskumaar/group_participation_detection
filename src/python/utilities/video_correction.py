import cv2
import numpy as np
import subprocess
import os

# === CONFIG ===
VIDEO_PATH = r"C:\Users\sures\Desktop\children_pilot_3_360.mp4"
OUTPUT_PATH_NO_AUDIO = r"C:\Users\sures\Desktop\children_pilot_3_360_corrected.mp4"
OUTPUT_PATH_FINAL = r"C:\Users\sures\Desktop\children_pilot_3_360_corrected_with_audio.mp4"

# Amount to shift in pixels
# Positive value moves image content LEFT (cut from right → attach left)
SHIFT_PX = 1000   # adjust until alignment looks correct

# === PROCESS VIDEO (NO AUDIO) ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open input video: {VIDEO_PATH}")

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH_NO_AUDIO, fourcc, fps, (W, H))

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Wrap horizontally
    shift_px = SHIFT_PX % W
    left = frame[:, :shift_px]
    right = frame[:, shift_px:]
    shifted = np.hstack((right, left))
    out.write(shifted)

    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"Processed {frame_idx} frames...")

cap.release()
out.release()
print(f"✅ Saved shifted video (no audio) → {OUTPUT_PATH_NO_AUDIO}")

# === MERGE VIDEO (SHIFTED) + AUDIO (ORIGINAL) ===
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Original video not found: {VIDEO_PATH}")
if not os.path.exists(OUTPUT_PATH_NO_AUDIO):
    raise FileNotFoundError(f"Shifted video not found: {OUTPUT_PATH_NO_AUDIO}")

cmd = [
    "ffmpeg",
    "-y",  # overwrite without asking
    "-i", OUTPUT_PATH_NO_AUDIO,  # shifted video (no audio)
    "-i", VIDEO_PATH,  # original video (has audio)
    "-c:v", "copy",  # copy video stream (no re-encoding)
    "-map", "0:v:0",  # take video from first input
    "-map", "1:a:0",  # take audio from second input
    "-shortest",  # stop when shorter stream ends
    OUTPUT_PATH_FINAL
]

print("Merging audio from original video...")
subprocess.run(cmd, check=True)
print(f"🎬 Done! Saved merged video with audio → {OUTPUT_PATH_FINAL}")
