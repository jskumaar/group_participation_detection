# capture_multi.py
import cv2
import os
import time
from pathlib import Path

# --- CONFIG ---
CAMERS = [1, 2, 3]  # replace with your camera indices or video device strings
OUT_DIR = Path("captures_aruco")  # will create captures/cam0, cam1, ...
IMG_FMT = "jpg"
WIDTH = 1280
HEIGHT = 720

# --- Setup output dirs ---
OUT_DIR.mkdir(exist_ok=True)
cam_dirs = []
for i in range(len(CAMERS)):
    d = OUT_DIR / f"cam{i}"
    d.mkdir(exist_ok=True)
    cam_dirs.append(d)

# --- Open captures ---
caps = []
for i, src in enumerate(CAMERS):
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {src} failed to open")
    caps.append(cap)

print("[INFO] Opened cameras:", CAMERS)
index = 0

try:
    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = 255 * np.ones((HEIGHT, WIDTH, 3), dtype='uint8')
            frames.append(frame)

        # Display previews tiled horizontally
        preview = cv2.hconcat(frames)
        cv2.putText(preview, f"Press 'c' to capture synced frame #{index}, 'q' to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("MultiCam Capture", preview)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('c'):
            for cam_idx, frame in enumerate(frames):
                outpath = cam_dirs[cam_idx] / f"img_{index:04d}.{IMG_FMT}"
                cv2.imwrite(str(outpath), frame)
            print(f"[SAVE] Captured frame index {index}")
            index += 1
        elif k == ord('q'):
            break
finally:
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
