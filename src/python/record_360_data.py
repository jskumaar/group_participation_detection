# calibrate_hybrid_with_record.py
import cv2, os, glob, argparse
from pathlib import Path
import numpy as np
import datetime
import time, sys


SAVE_DIR = Path("calib_data")
SAVE_DIR.mkdir(exist_ok=True)

CAMERA_INDEX        = 1
CAM_WIDTH           = 2880
CAM_HEIGHT          = 1440

# of the 360 camera
PREFERRED_NAMES = [
    "OBS Virtual Camera",
    "Insta360 Virtual Camera",
    "Insta360 Link",
    "Insta360",
    "OBS-Camera",
]

### camera find helpers ###
def _is_equirectangular(frame, ratio_tol=0.08):
    h, w = frame.shape[:2]
    ratio = w / float(h)
    return abs(ratio - 2.0) <= ratio_tol

def _probe_frame(cap, warmup=5):
    frame = None
    for _ in range(warmup):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
        else:
            time.sleep(0.02)
    return frame

def _open_dshow_by_name(name, width=None, height=None):
    cap = cv2.VideoCapture(f"video={name}", cv2.CAP_DSHOW)
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, frame = cap.read()
    if ok and frame is not None:
        return cap, frame
    cap.release()
    return None, None


def find_obs_or_insta_camera(preferred_names=PREFERRED_NAMES, width=None, height=None, prefer_equirect=True):
    system = sys.platform
    if system.startswith("win"):
        for name in preferred_names:
            cap, frame = _open_dshow_by_name(name, width, height)
            if cap is not None:
                return cap, "CAP_DSHOW(name)", name
    backends = []
    if system.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    elif system == "darwin":
        backends = [cv2.CAP_AVFOUNDATION]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    candidates = []
    for backend in backends:
        for idx in range(6):
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            if width and height:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            frame = _probe_frame(cap)
            if frame is None:
                cap.release()
                continue
            h, w = frame.shape[:2]
            candidates.append((cap, backend, idx, w, h, _is_equirectangular(frame)))
    if not candidates:
        return None, None, None
    if prefer_equirect:
        for cap, backend, idx, w, h, is_eq in candidates:
            if is_eq:
                return cap, f"backend={backend}", f"idx={idx} ({w}x{h})"
    cap, backend, idx, w, h, _ = candidates[0]
    return cap, f"backend={backend}", f"idx={idx} ({w}x{h})"

##############################


def record_video(cap, out_name="calib_session"):
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SAVE_DIR / f"{out_name}_{ts}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    print(f"[Recording] Saving video to {out_path}")
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow("Live (press q to stop)", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); out.release(); cv2.destroyAllWindows()
    return str(out_path)


def record_images(cap, out_name="calib_imgs", interval=30, N_images=40):
    """
    Save still images every N frames until 'q' is pressed or until N_images is reached.
    """
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = SAVE_DIR / f"{out_name}_{ts}"
    folder.mkdir(exist_ok=True)

    print(f"[Recording] Saving images to {folder}")
    frame_idx = 0; saved = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow("Live (press q to stop)", frame)

        if frame_idx % interval == 0:
            fname = folder / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(fname), frame)
            saved += 1
            print(f"Saved {fname}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1

        if saved >= N_images:
            print(f"Reached {N_images} images, stopping.")
            break

    cap.release(); cv2.destroyAllWindows()
    return str(folder)

def main():
    ap = argparse.ArgumentParser(description="Record calibration data and/or calibrate")
    ap.add_argument("--record_video", action="store_true", help="Record a calibration video")
    ap.add_argument("--record_images", action="store_true", help="Record calibration images")
    ap.add_argument("--interval", type=int, default=30, help="Frame interval for image capture")
    ap.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    args = ap.parse_args()

    cap, backend_label, source_label = find_obs_or_insta_camera(width=CAM_WIDTH, height=CAM_HEIGHT, prefer_equirect=True)


    if args.record_video:
        record_video(cap)
    elif args.record_images:
        record_images(cap, interval=args.interval)
    else:
        print("No recording option selected. Use --record_video or --record_images")

if __name__ == "__main__":
    main()
