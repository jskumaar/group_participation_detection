import cv2
import sys, time
import numpy as np

# ================================
# ArUco Setup
# ================================
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
det_params = cv2.aruco.DetectorParameters()
det_params.adaptiveThreshWinSizeMin = 3
det_params.adaptiveThreshWinSizeMax = 45
det_params.adaptiveThreshWinSizeStep = 10
det_params.minMarkerPerimeterRate = 0.01
det_params.maxMarkerPerimeterRate = 4.0
det_params.polygonalApproxAccuracyRate = 0.03
det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

detector = cv2.aruco.ArucoDetector(ARUCO_DICT, det_params)

# High-resolution capture
CAM_WIDTH  = 2880
CAM_HEIGHT = 1440

# Optional: load calibration for pose estimation
try:
    calib = np.load("aruco_calib_2x3.npz")
    camera_matrix = calib["camera_matrix"]
    dist_coeffs = calib["dist_coeffs"]
    print("[INFO] Loaded camera calibration.")
except Exception:
    camera_matrix = None
    dist_coeffs = None
    print("[WARN] No calibration file found — pose estimation disabled.")


# ================================
# Helper functions for camera search
# ================================
PREFERRED_NAMES = [
    "OBS Virtual Camera",
    "Insta360 Virtual Camera",
    "Insta360 Link",
    "Insta360",
    "OBS-Camera",
]

def _open_dshow_by_name(name, width=None, height=None):
    cap = cv2.VideoCapture(f"video={name}", cv2.CAP_DSHOW)
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, frame = cap.read()
    if ok and frame is not None:
        return cap, frame
    cap.release()
    return None, None

def _probe_frame(cap, warmup=5):
    frame = None
    for _ in range(warmup):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
        else:
            time.sleep(0.02)
    return frame

def _is_equirectangular(frame, ratio_tol=0.08):
    h, w = frame.shape[:2]
    ratio = w / float(h)
    return abs(ratio - 2.0) <= ratio_tol

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


# ================================
# Initialize Camera
# ================================
cap, backend_label, source_label = find_obs_or_insta_camera(width=CAM_WIDTH, height=CAM_HEIGHT)
if cap is None:
    print("[WARN] No Insta/OBS camera found. Using default webcam.")
    cap = cv2.VideoCapture(0)
    backend_label, source_label = "default", "0"

if not cap.isOpened():
    raise RuntimeError("No usable camera found.")

print(f"[INFO] Using camera: {source_label} via {backend_label}")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

print("[INFO] Press 'q' to quit.")


# ================================
# Main loop
# ================================
while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Detect ArUco marker(s) ---
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i, marker_id in enumerate(ids.flatten()):
            c = corners[i][0]
            center = np.mean(c, axis=0).astype(int)
            cv2.circle(frame, tuple(center), 6, (0, 255, 0), -1)
            cv2.putText(frame, f"ID: {marker_id}",
                        (center[0] + 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- Pose estimation (if calibration available) ---
            if camera_matrix is not None and dist_coeffs is not None:
                marker_length = 0.05  # meters
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[i]], marker_length, camera_matrix, dist_coeffs)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                pos = tvec[0][0]
                cv2.putText(frame, f"Pos: {pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}m",
                            (center[0]-60, center[1]+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "No marker detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Single ArUco Tag Detection", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
