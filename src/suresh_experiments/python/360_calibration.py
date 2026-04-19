# pose_from_equirect_hybrid.py
# Estimate board pose from a stitched equirectangular frame by:
# 1) detecting markers on equirect,
# 2) warping a local gnomonic (perspective) view,
# 3) running standard aruco pose on the perspective view.

import math, argparse, os, glob
from pathlib import Path
import numpy as np
import cv2
import cv2.aruco as aruco


# import your board builder here (same as before)
from generate_calibration_boards import build_hybrid_board  # assumes you saved earlier board code

# ------------ Hybrid A3 board geometry (must match your printed board) ------------
DPI = 300
W_MM, H_MM = 297.0, 420.0     # A3 portrait
MARGIN_MM = 10.0
DICT = cv2.aruco.DICT_5X5_1000

BIG_MARKER_MM  = 90.0         # corner markers size
EDGE_MARKER_MM = 60.0         # edge markers size
EDGE_OFFSET_MM = 30.0         # offset from paper edge for mid-edge markers

CORNER_IDS = [0, 1, 2, 3]     # TL, TR, BL, BR
EDGE_IDS   = [10, 11, 12, 13] # top, bottom, left, right
# -------------------------------------------------------------------------------

def mm_to_m(x): return x * 1e-3


# ---------- Equirect helpers ----------
def px_to_dir_equirect(u, v, W, H):
    # u in [0,W), v in [0,H)
    lon = (u / float(W)) * 2.0 * math.pi - math.pi        # [-pi, pi]
    lat = math.pi/2.0 - (v / float(H)) * math.pi          # [pi/2,-pi/2] top->bottom
    x = math.cos(lat) * math.cos(lon)
    y = math.sin(lat)
    z = math.cos(lat) * math.sin(lon)
    return np.array([x, y, z], dtype=np.float32)

def dir_to_px_equirect(d, W, H):
    x, y, z = d
    lon = math.atan2(z, x)
    lat = math.asin(y)
    u = (lon + math.pi) / (2.0 * math.pi) * W
    v = (math.pi/2.0 - lat) / math.pi * H
    return u, v

def build_camera_basis(forward, up_hint=np.array([0,1,0], np.float32)):
    f = forward / np.linalg.norm(forward)
    r = np.cross(f, up_hint); n = np.linalg.norm(r)
    if n < 1e-6:
        up_hint = np.array([0,0,1], np.float32)
        r = np.cross(f, up_hint); n = np.linalg.norm(r)
    r /= n
    u = np.cross(r, f)
    return r, u, f   # right, up, forward

def gnomonic_sample(equi, center_dir, fov_deg=90.0, out_w=1280, out_h=720):
    """Create a perspective (gnomonic) view centered at center_dir with square pixels."""
    H, W = equi.shape[:2]
    r, u, f = build_camera_basis(center_dir)
    fov = math.radians(fov_deg)
    # pinhole focal in pixels for out_w across fov
    fx = fy = 0.5 * out_w / math.tan(0.5 * fov)
    cx, cy = out_w * 0.5, out_h * 0.5

    ys, xs = np.indices((out_h, out_w), dtype=np.float32)
    x_cam = (xs - cx) / fx
    y_cam = (ys - cy) / fy
    z_cam = np.ones_like(x_cam)

    # ray in local camera coords -> world dir
    norm = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
    x_cam /= norm; y_cam /= norm; z_cam /= norm
    # world dir = x_cam*r + y_cam*u + z_cam*f
    dir_world = (x_cam[...,None]*r + y_cam[...,None]*u + z_cam[...,None]*f)

    # map to equirect UV
    xw, yw, zw = dir_world[...,0], dir_world[...,1], dir_world[...,2]
    lon = np.arctan2(zw, xw)     # [-pi, pi]
    lat = np.arcsin(yw)          # [-pi/2, pi/2]
    u_e = (lon + math.pi) / (2.0*math.pi) * W
    v_e = (math.pi/2.0 - lat) / math.pi * H

    # sample with bilinear interpolation
    map_x = u_e.astype(np.float32)
    map_y = v_e.astype(np.float32)
    persp = cv2.remap(equi, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    # return perspective image and intrinsic matrix for that synthetic view
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    return persp, K

# ---------- Detection & pose ----------

def get_board(board_type="hybrid"):
    if board_type == "hybrid":
        from generate_calibration_boards import build_hybrid_board
        return build_hybrid_board()
    
    elif board_type == "grid":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        board = aruco.GridBoard(
            (2, 3),    # number of rows, columns
            0.1145,  # marker length in meters
            0.023,   # marker separation in meters
            aruco_dict
        )
        return board, aruco_dict

    else:
        raise ValueError(f"Unknown board type: {board_type}")

def detect_markers(img, dictionary):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    return corners, ids

def mean_direction_from_detections(corners, W, H):
    # use centers of each detected marker in equirect to estimate a forward direction
    dirs = []
    for c in corners:
        cx = np.mean(c[0][:,0]); cy = np.mean(c[0][:,1])
        d = px_to_dir_equirect(cx, cy, W, H)
        d = d / np.linalg.norm(d)
        dirs.append(d)
    if not dirs:
        return None
    dmean = np.mean(np.stack(dirs,0), axis=0)
    n = np.linalg.norm(dmean)
    return dmean / (n + 1e-9)

def estimate_pose_from_perspective(persp_bgr, K, board, dictionary, dist=None, min_markers=4, draw=False):
    corners, ids = detect_markers(persp_bgr, dictionary)
    if ids is None or len(ids) < min_markers:
        return None, None, None, corners, ids
    rvec, tvec = None, None
    ok, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, K, np.zeros((5,1), np.float32) if dist is None else dist, rvec, tvec)
    if ok > 0 and draw:
        out = persp_bgr.copy()
        cv2.aruco.drawDetectedMarkers(out, corners, ids)
        cv2.drawFrameAxes(out, K, np.zeros((5,)), rvec, tvec, 0.1)  # 0.1 m axis
        cv2.imshow("Perspective pose", out); cv2.waitKey(1)
    return rvec, tvec, ok, corners, ids



def auto_fov_from_bbox(bbox, center_px, W, H, out_w, out_h, margin_deg=10.0):
    """
    Estimate a good horizontal FoV for gnomonic projection based on how
    large the board appears in the equirectangular image.

    Parameters
    ----------
    bbox : (x1, y1, x2, y2) 
        Bounding box around the detected board (pixels in equirect image).
    center_px : (u, v)
        Center of the board in equirect (pixels).
    W, H : int
        Width and height of the equirectangular frame (pixels).
    out_w, out_h : int
        Desired gnomonic crop size (pixels).
    margin_deg : float
        Extra margin in degrees to avoid clipping.

    Returns
    -------
    fov_deg : float
        Suggested horizontal FoV (degrees).
    """
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    # board angular width in longitude
    board_ang_w = (bw / float(W)) * 360.0
    # board angular height in latitude
    board_ang_h = (bh / float(H)) * 180.0

    # match aspect ratio of the crop
    aspect = out_h / float(out_w)
    hfov_from_h = math.degrees(
        2 * math.atan(math.tan(math.radians(board_ang_h)/2) / aspect)
    )

    # choose the larger to guarantee fit, add margin
    hfov = max(board_ang_w, hfov_from_h) + margin_deg

    # clamp to safe range
    hfov = max(40.0, min(120.0, hfov))
    return hfov


def process_frame_equi(equi_bgr, board, dictionary,
                       out_w=1280, out_h=720,
                       min_markers=4, show_debug=True, save_debug=True, frame_name="frame"):
    H, W = equi_bgr.shape[:2]

    # 1) detect markers directly on equirect
    corners_eq, ids_eq = detect_markers(equi_bgr, dictionary)
    if ids_eq is None or len(ids_eq) < min_markers:
        print(f"[{frame_name}] Not enough detections in equirect ({0 if ids_eq is None else len(ids_eq)} markers)")
        return None, None, None

    # bounding box
    xs, ys = [], []
    for c in corners_eq:
        xs.extend(c[0][:,0]); ys.extend(c[0][:,1])
    bbox = (min(xs), min(ys), max(xs), max(ys))
    center_px = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)

    # auto FoV
    fov_deg = auto_fov_from_bbox(bbox, center_px, W, H, out_w, out_h, margin_deg=10.0)
    print(f"[{frame_name}] Equirect markers={len(ids_eq)}  bbox={bbox}  auto-FoV={fov_deg:.1f}°")

    # 2) make gnomonic crop
    center_dir = mean_direction_from_detections(corners_eq, W, H)
    persp, K = gnomonic_sample(equi_bgr, center_dir, fov_deg=fov_deg, out_w=out_w, out_h=out_h)

    # --- SAVE/SHOW perspective view regardless of detection ---
    debug_path = f"gnomonic_{frame_name}.jpg"
    cv2.imwrite(debug_path, persp)
    if show_debug:
        cv2.imshow("Perspective crop", persp)
        cv2.waitKey(1)
    print(f"[{frame_name}] Saved perspective crop → {debug_path}")

    # 3) detect markers on perspective crop
    corners_p, ids_p = detect_markers(persp, dictionary)
    if ids_p is None or len(ids_p) < min_markers:
        print(f"[{frame_name}] Pose FAILED in perspective (markers={0 if ids_p is None else len(ids_p)})")
        return None, None, None

    # draw detected markers
    debug_view = persp.copy()
    cv2.aruco.drawDetectedMarkers(debug_view, corners_p, ids_p)
    cv2.imwrite(f"gnomonic_{frame_name}_det.jpg", debug_view)

    # 4) estimate pose
    ok, rvec, tvec = cv2.aruco.estimatePoseBoard(corners_p, ids_p, board, K, np.zeros((5,1)))
    if ok > 0:
        cv2.drawFrameAxes(debug_view, K, np.zeros((5,)), rvec, tvec, 0.1)
        cv2.imwrite(f"gnomonic_{frame_name}_pose.jpg", debug_view)
        if show_debug:
            cv2.imshow("Pose", debug_view); cv2.waitKey(1)
        print(f"[{frame_name}] Pose OK | {len(ids_p)} markers")
    else:
        print(f"[{frame_name}] Pose FAILED even after perspective markers detected")

    return rvec, tvec, K


def main():
    ap = argparse.ArgumentParser(description="Estimate hybrid A3 board pose from stitched equirect frames.")
    ap.add_argument("--images", nargs="+", default=["equi/*.jpg", "equi/*.png"], help="Glob(s) for equirect frames.")
    ap.add_argument("--video", type=str, default=None, help="Optional video path.")
    ap.add_argument("--fov", type=float, default=90.0, help="Perspective FOV for gnomonic view.")
    ap.add_argument("--size", type=str, default="1280x720", help="Perspective size WxH.")
    ap.add_argument("--debug", action="store_true", help="Show perspective pose window.")
    ap.add_argument("--board", type=str, default="grid",
                choices=["hybrid", "grid"],
                help="Which board type to use: hybrid (A3 custom), grid (ArUco GridBoard)")

    args = ap.parse_args()

    out_w, out_h = map(int, args.size.lower().split("x"))
    board, dictionary = get_board(args.board)


    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened(): raise RuntimeError(f"Cannot open {args.video}")
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            rvec, tvec, K = process_frame_equi(
                frame, board, dictionary,
                out_w=out_w, out_h=out_h,
                show_debug=args.debug,
                frame_name=f"frame{frame_idx:04d}"
            )
            if rvec is not None:
                print("Pose OK | rvec:", rvec.ravel(), " tvec:", tvec.ravel())
            if args.debug:
                cv2.imshow("Equirect", frame)
                if cv2.waitKey(1) & 0xFF == 27: break

            frame_idx += 1
        cap.release()
        cv2.destroyAllWindows()
    else:
        paths = []
        for pat in args.images:
            paths.extend(glob.glob(pat))
        paths = sorted(paths)
        if not paths:
            print("No images found."); return
        for p in paths:
            equi = cv2.imread(p)
            if equi is None: 
                print(f"[skip] {p}")
                continue
            frame_name = Path(p).stem
            rvec, tvec, K = process_frame_equi(
                equi, board, dictionary,
                out_w=out_w, out_h=out_h,
                show_debug=args.debug,
                frame_name=frame_name
            )
            if rvec is not None:
                print(f"{Path(p).name}: Pose OK | rvec {rvec.ravel()}  tvec {tvec.ravel()}")
            else:
                print(f"{Path(p).name}: Pose FAILED")

if __name__ == "__main__":
    main()
