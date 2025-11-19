import cv2
import sys, time
import numpy as np

# ArUco board parameters
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)

BOARD_ROWS    = 3   # rows of markers
BOARD_COLS    = 2   # cols of markers
MARKER_LENGTH = 0.1145   # meters (adjust to your printed marker size)
MARKER_SEP    = 0.023   # meters (gap between markers)

# board = cv2.aruco.GridBoard_create(
#     markersX=BOARD_COLS,
#     markersY=BOARD_ROWS,
#     markerLength=MARKER_LENGTH,
#     markerSeparation=MARKER_SEP,
#     dictionary=ARUCO_DICT
# )

board = cv2.aruco.GridBoard(
    (BOARD_COLS, BOARD_ROWS),
    MARKER_LENGTH,
    MARKER_SEP,
    ARUCO_DICT
)

CAM_WIDTH  = 2880
CAM_HEIGHT = 1440

all_corners = []
all_ids = []
image_size = None

# --- Camera init ---

PREFERRED_NAMES = [
    "OBS Virtual Camera",
    "Insta360 Virtual Camera",
    "Insta360 Link",
    "Insta360",
    "OBS-Camera",
]


###############  Helper Functions ###############

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

################################

# cap, backend_label, source_label = find_obs_or_insta_camera(width=CAM_WIDTH, height=CAM_HEIGHT, prefer_equirect=True)

cap = cv2.VideoCapture(0)  # Default camera


if cap is None:
    raise RuntimeError("No usable camera found.")
# print(f"[INFO] Using camera: {source_label} via {backend_label}")


cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

det_params = cv2.aruco.DetectorParameters()
det_params.adaptiveThreshWinSizeMin = 3
det_params.adaptiveThreshWinSizeMax = 45
det_params.adaptiveThreshWinSizeStep = 10
det_params.minMarkerPerimeterRate = 0.01
det_params.maxMarkerPerimeterRate = 4.0
det_params.polygonalApproxAccuracyRate = 0.03
det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

print("[INFO] Press 'c' to capture a frame, 'u' to undo last, 'q' to finish.")

stable_count = 0
STABLE_FRAMES_REQUIRED = 5   # how many frames in a row
MIN_MARKERS_REQUIRED   = 4   # need at least 4 markers detected

while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=det_params)

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # --- Auto-capture when stable ---
        if len(ids) >= MIN_MARKERS_REQUIRED:
            stable_count += 1
            if stable_count >= STABLE_FRAMES_REQUIRED:
                all_corners.append(corners)
                all_ids.append(ids)
                image_size = gray.shape[::-1]
                print(f"[INFO] Auto-captured frame #{len(all_corners)}")
                stable_count = 0
                time.sleep(0.5)  # pause to avoid rapid duplicate captures
        else:
            stable_count = 0
    else:
        stable_count = 0

    # HUD
    cv2.putText(frame, f"Captured: {len(all_corners)}  [c=manual, u=undo, q=done]",
                (20, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("ArUco Board Calibration", frame)
    k = cv2.waitKey(1) & 0xFF

    # manual capture
    if k == ord('c'):
        if ids is not None and len(ids) > 0:
            all_corners.append(corners)
            all_ids.append(ids)
            image_size = gray.shape[::-1]
            print(f"[INFO] Captured {len(all_corners)} views (manual).")
        else:
            print("[WARN] No markers detected in this frame.")

    elif k == ord('u'):
        if all_corners:
            all_corners.pop(); all_ids.pop()
            print(f"[INFO] Undo. Captures now: {len(all_corners)}")

    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if len(all_corners) < 10:
    raise RuntimeError(f"Only {len(all_corners)} valid captures. Collect at least ~20.")

# --- Calibrate using ArUco ---
print("[INFO] Calibrating…")
rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(
    all_corners, all_ids, board, image_size, None, None
)

print(f"[RESULT] RMS reprojection error: {rms:.4f}")
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs.ravel())

np.savez("aruco_calib_2x3.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
         rms=rms, image_size=image_size)
print("[INFO] Saved to aruco_calib_2x3.npz")