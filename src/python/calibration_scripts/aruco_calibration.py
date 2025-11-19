import cv2
import numpy as np
import glob
import os

# =======================
# CONFIGURATION
# =======================
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Grid of markers: columns x rows
MARKERS_X = 3
MARKERS_Y = 2

# Marker size and spacing (meters)
MARKER_LENGTH = 0.1145
MARKER_SEPARATION = 0.023

# Folder with calibration images per camera
BASE_FOLDER = "captures_aruco"
CAM_DIRS = [f"{BASE_FOLDER}/cam0", f"{BASE_FOLDER}/cam1", f"{BASE_FOLDER}/cam2"]

# Output
OUT_FILE = "multicam_aruco_calib.npz"

# =======================
# CREATE GRID BOARD
# =======================
board = cv2.aruco.GridBoard(
    (MARKERS_X, MARKERS_Y),
    MARKER_LENGTH,
    MARKER_SEPARATION,
    ARUCO_DICT
)
print(f"[INFO] Using {MARKERS_X}x{MARKERS_Y} ArUco grid")

# =======================
# DETECTION PARAMETERS
# =======================
det_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(ARUCO_DICT, det_params)

# =======================
# CALIBRATION PER CAMERA
# =======================
all_intrinsics = []
all_dist_coeffs = []
all_rvecs = []
all_tvecs = []
image_size = None

for cam_dir in CAM_DIRS:
    print(f"\n[INFO] Processing {cam_dir}...")
    image_files = sorted(glob.glob(os.path.join(cam_dir, "*.jpg")))
    if not image_files:
        print(f"[WARN] No images found in {cam_dir}, skipping...")
        continue

    all_corners = []
    all_ids = []
    counter = []

    for f in image_files:
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            # corners is currently a list of arrays -> convert to single numpy array
            corners_array = np.array(corners, dtype=np.float32)
            ids_array     = np.array(ids, dtype=np.int32)

            all_corners.append(corners_array)
            all_ids.append(ids_array)
            counter.append(len(ids))
            image_size = gray.shape[::-1]


    # --- Calibrate ---
    rms, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(
        corners=all_corners,
        ids=all_ids,
        counter=counter,      # <-- this fixes your error
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    print(f"[RESULT] RMS reprojection error: {rms:.4f}")
    print("Camera matrix:\n", K)
    print("Distortion coefficients:\n", D.ravel())

    all_intrinsics.append(K)
    all_dist_coeffs.append(D)
    all_rvecs.append(rvecs)
    all_tvecs.append(tvecs)

# =======================
# SAVE RESULTS
# =======================
np.savez(
    OUT_FILE,
    intrinsics=all_intrinsics,
    dist_coeffs=all_dist_coeffs,
    rvecs=all_rvecs,
    tvecs=all_tvecs,
    image_size=image_size
)

print(f"\n✅ Saved all calibration results to {OUT_FILE}")
