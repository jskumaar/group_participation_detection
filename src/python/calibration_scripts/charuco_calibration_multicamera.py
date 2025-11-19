# multicam_calibrate.py
import cv2
import numpy as np
import os
from pathlib import Path

# ---------------- CONFIG ----------------
CAPTURE_DIR = Path("captures")
N_CAMERAS = len([p for p in CAPTURE_DIR.iterdir() if p.is_dir()])  # expects cam0, cam1, ...
REF_CAM = 0  # reference camera index (extrinsics will be expressed relative to this cam)
MIN_CHARUCO_CORNERS = 10

# Charuco board definition - must match the board used in capture!
CHARUCOBOARD_ROWCOUNT = 7
CHARUCOBOARD_COLCOUNT = 5
SQUARE_LENGTH = 0.036  # meters (example)
MARKER_LENGTH = 0.029  # meters (example)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
board = cv2.aruco.CharucoBoard((CHARUCOBOARD_COLCOUNT, CHARUCOBOARD_ROWCOUNT),
                               SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT)

# Detection parameters (tweak if needed)
det_params = cv2.aruco.DetectorParameters()
det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

# ---------------- helper functions ----------------
def load_image_list(cam_idx):
    d = CAPTURE_DIR / f"cam{cam_idx}"
    imgs = sorted([str(p) for p in d.iterdir() if p.suffix.lower() in (".jpg",".png",".jpeg")])
    return imgs

def detect_charuco(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None, None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=det_params)
    if ids is None or len(ids) == 0:
        return None, None, None
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    return charuco_corners, charuco_ids, img.shape[:2][::-1]  # image_size (w,h)

# ---------------- gather detections per camera per view ----------------
# Expect same number of images per camera and matching indices
imgs_per_cam = [load_image_list(i) for i in range(N_CAMERAS)]
n_views = min(len(lst) for lst in imgs_per_cam)
print(f"[INFO] Found {N_CAMERAS} cameras, {n_views} synchronized views (using min per-camera count)")

# Store results: per-camera per-view charuco_corners/ids
per_cam_detections = [[None]*n_views for _ in range(N_CAMERAS)]
image_size = None

for cam in range(N_CAMERAS):
    for v in range(n_views):
        charuco_corners, charuco_ids, img_size = detect_charuco(imgs_per_cam[cam][v])
        if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) >= MIN_CHARUCO_CORNERS:
            per_cam_detections[cam][v] = (charuco_corners, charuco_ids)
            image_size = img_size  # last valid
        else:
            per_cam_detections[cam][v] = None

# Count valid views per camera
for cam in range(N_CAMERAS):
    valid = sum(1 for v in range(n_views) if per_cam_detections[cam][v] is not None)
    print(f"[INFO] Camera {cam} valid views: {valid}/{n_views}")

# ---------------- individual calibrations ----------------
intrinsics = {}
for cam in range(N_CAMERAS):
    all_charuco_corners = []
    all_charuco_ids = []
    for v in range(n_views):
        det = per_cam_detections[cam][v]
        if det is None:
            continue
        corners, ids = det
        all_charuco_corners.append(corners)
        all_charuco_ids.append(ids)

    if len(all_charuco_corners) < 5:
        raise RuntimeError(f"Camera {cam} has too few Charuco views ({len(all_charuco_corners)}). Collect more captures.")
    print(f"[INFO] Calibrating camera {cam} with {len(all_charuco_corners)} views...")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners, all_charuco_ids, board, image_size, None, None
    )
    print(f"  -> Camera {cam} RMS: {rms:.4f}")
    intrinsics[cam] = dict(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rms=rms)
    np.savez(f"cam{cam}_intrinsics.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rms=rms, image_size=image_size)
    print(f"[SAVE] cam{cam}_intrinsics.npz")

# ---------------- stereo / extrinsics per camera -> REF_CAM ----------------
# We'll produce extrinsics (R, T) that map points from camera_i to camera_ref:
# X_ref = R_i_to_ref * X_i + T_i_to_ref

extrinsics = {}
ref_K = intrinsics[REF_CAM]['camera_matrix']
ref_dist = intrinsics[REF_CAM]['dist_coeffs']

for cam in range(N_CAMERAS):
    if cam == REF_CAM:
        extrinsics[cam] = (np.eye(3), np.zeros((3,1)))
        continue

    obj_points = []   # list of (N,3) arrays of board 3D points in board coords
    img_points_ref = []  # list of (N,2) arrays for ref cam
    img_points_cam = []  # list of (N,2) arrays for this cam

    K_cam = intrinsics[cam]['camera_matrix']
    dist_cam = intrinsics[cam]['dist_coeffs']

    for v in range(n_views):
        det_ref = per_cam_detections[REF_CAM][v]
        det_cam = per_cam_detections[cam][v]
        if det_ref is None or det_cam is None:
            continue

        corners_ref, ids_ref = det_ref
        corners_cam, ids_cam = det_cam

        # ids are 1D arrays shaped (N,1) -> flatten
        ids_ref_f = ids_ref.flatten()
        ids_cam_f = ids_cam.flatten()

        # find intersection of ids present in both views
        common_ids, idx_ref, idx_cam = np.intersect1d(ids_ref_f, ids_cam_f, return_indices=True)

        if len(common_ids) < MIN_CHARUCO_CORNERS:
            continue

        # board.chessboardCorners has 3D coordinates (N_all, 3) in meters for all charuco corners
        # use the common ids to pick the corresponding 3D points
        # Note: ensure board.chessboardCorners is numpy array shape (N,3)
        board_corners_3d = np.array(board.chessboardCorners)  # shape (N_all, 3)
        objp = board_corners_3d[common_ids]  # pick by id (ids are indexes into board corners)
        # gather corresponding image points
        imgp_ref = corners_ref[idx_ref].reshape(-1,2)
        imgp_cam = corners_cam[idx_cam].reshape(-1,2)

        obj_points.append(objp.astype(np.float64))
        img_points_ref.append(imgp_ref.astype(np.float64))
        img_points_cam.append(imgp_cam.astype(np.float64))

    if len(obj_points) < 5:
        raise RuntimeError(f"Not enough overlapping views between camera {REF_CAM} and camera {cam} for stereo calibration ({len(obj_points)}).")

    # Prepare data for stereoCalibrate: lists of arrays shaped (N,1,3) and (N,1,2)
    obj_points_s = [op.reshape(-1,1,3) for op in obj_points]
    img_points_ref_s = [ip.reshape(-1,1,2) for ip in img_points_ref]
    img_points_cam_s = [ip.reshape(-1,1,2) for ip in img_points_cam]

    # stereoCalibrate with fixed intrinsics (we already calibrated each camera)
    flags = cv2.CALIB_FIX_INTRINSIC
    print(f"[INFO] Running stereoCalibrate between cam{REF_CAM} and cam{cam} using {len(obj_points_s)} common views...")
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    retval, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_points_s,            # object points (board 3D)
        img_points_ref_s,        # image points in ref cam
        img_points_cam_s,        # image points in this cam
        ref_K,                   # cameraMatrix1
        ref_dist,                # distCoeffs1
        K_cam,                   # cameraMatrix2
        dist_cam,                # distCoeffs2
        image_size,
        flags=flags,
        criteria=criteria
    )
    print(f"  -> stereo RMS: {retval:.6f}")
    # R,T transform from cam to ref:
    # A point X in cam coords can be transformed to ref coords by: X_ref = R * X_cam + T
    extrinsics[cam] = (R, T)
    np.savez(f"extrinsics_cam{cam}_to_cam{REF_CAM}.npz", R=R, T=T, E=E, F=F, rms=retval)
    print(f"[SAVE] extrinsics_cam{cam}_to_cam{REF_CAM}.npz")

print("[DONE] All intrinsics and extrinsics saved.")
