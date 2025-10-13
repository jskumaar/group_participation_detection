#!/usr/bin/env python3
import cv2, numpy as np, math, os, time, csv
from datetime import datetime
from ultralytics import YOLO
import mediapipe as mp
import sys
from collections import deque


# =========================
# CONFIG
# =========================
USE_CAMERA = False                 # Set to False to read from file
VIDEO_FILE_PATH = r"G:\My Drive\Research\(Shared) Project_mixed_group_interaction\Pilot_data_collection\Session_1\Pilot_1_360_hd_interaction.mp4"    # Path to mp4 if not using camera


CSV_BUFFER = []
CSV_FLUSH_INTERVAL = 50   # write every 50 frames (tune as needed)

COLORS = [(0,0,255), (0,255,0), (255,0,0), (0,255,255)]


YOLO_INTERVAL_SEC   = 10.0
POSE_INTERVAL_SEC   = 1.0

CAMERA_INDEX        = 1
CAM_WIDTH           = 2880
CAM_HEIGHT          = 1440

YOLO_MODEL_PATH     = "yolov8n.pt"
YOLO_CONF_THRESHOLD = 0.35
YOLO_IMGSZ          = 960

PROJECTION_MODE     = "gnomonic"
GNOMONIC_OUT_W      = 512
GNOMONIC_OUT_H      = 512
GNOMONIC_ROLL_DEG   = 180.0

GAZE_MAX_DEG = 8.0
GAZE_FRONT_COS = 0.0
USE_ROBOT = True
ROBOT_LONLAT = (0.0, 0.0)
CALIB_YAW_DEG  = 0.0
CALIB_ROLL_DEG = 0.0

SAVE_DIR            = "live_out"
SHOW_EQUIRECT_WINDOW = True
EQUIRECT_DISPLAY_MAX_W = 1280

# New toggles
SHOW_FACE_BBOX       = True
SHOW_PERSON_BBOX     = True
SAVE_COMPARISONS     = False
SAVE_EQUIRECT_PREVIEW = True

# =========================
# FaceMesh + PnP
# =========================
FACE_3D_IDXS = [1, 152, 33, 263, 61, 291]
FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1)
], dtype=np.float64)

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

PREFERRED_NAMES = [
    "OBS Virtual Camera",
    "Insta360 Virtual Camera",
    "Insta360 Link",
    "Insta360",
    "OBS-Camera",
]


GAZE_HISTORY = deque(maxlen=30)   # average over past 30 frames (adjust N)


# ---------- camera find helpers ----------
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


# ---------- math helpers ----------
def _dir_from_lonlat(lam, phi):
    c = np.cos(phi)
    return np.array([c*np.cos(lam), np.sin(phi), c*np.sin(lam)], dtype=np.float64)

def _basis_from_forward(f):
    f = f/np.linalg.norm(f)
    up = np.array([0, 1, 0.0])
    u = up - f*np.dot(up, f)
    if np.linalg.norm(u) < 1e-9:
        up = np.array([1, 0, 0.0])
        u = up - f*np.dot(up, f)
    u = u/np.linalg.norm(u)
    r = np.cross(u, f)
    r = r/np.linalg.norm(r)
    return r, u, f

def _apply_roll(r, u, roll_rad):
    c, s = math.cos(roll_rad), math.sin(roll_rad)
    r2 = c*r + s*u
    u2 = c*u - s*r
    return r2, u2

def _lonlat_from_px(u, v, W, H):
    lam = 2*math.pi*(u/W) - math.pi
    phi = math.pi/2 - math.pi*(v/H)
    return lam, phi

def dir_to_equirect_px(D, W, H):
    Dx, Dy, Dz = D
    lam = math.atan2(Dz, Dx)
    phi = math.asin(np.clip(Dy, -1.0, 1.0))
    u = (lam + math.pi) / (2 * math.pi) * W
    v = (math.pi/2 - phi) / math.pi * H
    return int(round(u)), int(round(v))

def crop_px_to_global_dir(px, py, crop_w, crop_h, fov_deg, r_vec, u_vec, f_vec):
    x_ndc = (px / crop_w - 0.5) * 2
    y_ndc = (py / crop_h - 0.5) * 2
    hfov = math.radians(fov_deg)
    vfov = 2 * math.atan(math.tan(hfov/2) * (crop_h/crop_w))
    lx = math.tan(hfov/2) * x_ndc
    ly = math.tan(vfov/2) * y_ndc
    lz = 1.0
    dir_cam = np.array([lx, ly, lz])
    dir_cam /= np.linalg.norm(dir_cam)
    return _unit(r_vec*dir_cam[0] + u_vec*dir_cam[1] + f_vec*dir_cam[2])

def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / (n + 1e-9)

def _ang_deg(a, b):
    a = _unit(a); b = _unit(b)
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))

# ---------- pose estimation ----------
def get_camera_matrix(w, h):
    focal_length = w
    center = (w / 2, h / 2)
    return np.array([[focal_length, 0, center[0]],
                     [0, focal_length, center[1]],
                     [0, 0, 1]], dtype="double")

def estimate_head_pose_with_visuals(bgr):
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    lms = results.multi_face_landmarks[0]
    xs, ys = [], []
    for lm in lms.landmark:
        xs.append(int(lm.x * w)); ys.append(int(lm.y * h))
    head_box = (min(xs), min(ys), max(xs), max(ys))
    face_2d = []
    for idx in FACE_3D_IDXS:
        lm = lms.landmark[idx]
        face_2d.append((lm.x * w, lm.y * h))
    face_2d = np.array(face_2d, dtype=np.float64)
    cam_matrix = get_camera_matrix(w, h)
    dist = np.zeros((4, 1), dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(FACE_3D_MODEL, face_2d, cam_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = [math.degrees(a) for a in angles]
    axis_len = 100.0
    nose_end_3d = np.array([(0.0, 0.0, axis_len)], dtype=np.float64)
    nose_end_2d, _ = cv2.projectPoints(nose_end_3d, rvec, tvec, cam_matrix, dist)
    nose_tip_2d = (int(face_2d[0, 0]), int(face_2d[0, 1]))
    nose_end_2d = tuple(map(int, nose_end_2d[0].ravel()))
    return pitch, yaw, roll, head_box, nose_tip_2d, nose_end_2d, rvec

# ---------- equirectangular helpers ----------
def equirect_to_gnomonic(equi_bgr, *, center_px=None, center_lonlat=None,
                         yaw0_deg=0.0, roll_deg=0.0, fov_deg=80.0,
                         out_w=512, out_h=512):
    H, W = equi_bgr.shape[:2]
    if center_px is not None:
        lam, phi = _lonlat_from_px(center_px[0], center_px[1], W, H)
    elif center_lonlat is not None:
        lam = math.radians(center_lonlat[0]); phi = math.radians(center_lonlat[1])
    else:
        raise ValueError("Provide center_px or center_lonlat")

    lam += math.radians(yaw0_deg)
    fwd = _dir_from_lonlat(lam, phi)
    r, u, f = _basis_from_forward(fwd)
    r, u = _apply_roll(r, u, math.radians(roll_deg))

    hfov = math.radians(fov_deg)
    vfov = 2*math.atan(math.tan(hfov/2) * (out_h/out_w))

    xs = np.linspace(-math.tan(hfov/2), math.tan(hfov/2), out_w, dtype=np.float64)
    ys = np.linspace(-math.tan(vfov/2), math.tan(vfov/2), out_h, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys); Z = np.ones_like(X)

    denom = np.sqrt(X*X + Y*Y + Z*Z)
    lx, ly, lz = X/denom, Y/denom, Z/denom

    Dx = r[0]*lx + u[0]*ly + f[0]*lz
    Dy = r[1]*lx + u[1]*ly + f[1]*lz
    Dz = r[2]*lx + u[2]*ly + f[2]*lz

    lam = np.arctan2(Dz, Dx)
    phi = np.arcsin(np.clip(Dy, -1.0, 1.0))
    mapx = ((lam + math.pi) / (2*math.pi) * W).astype(np.float32)
    mapy = ((math.pi/2 - phi) / math.pi * H).astype(np.float32)

    crop = cv2.remap(equi_bgr, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)
    return crop

def fov_from_bbox_exact(bbox, center_px, W, H, out_w, out_h):
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    _, phi_c = _lonlat_from_px(center_px[0], center_px[1], W, H)
    ang_w = (bw / W) * (2 * math.pi) * math.cos(phi_c)
    ang_h = (bh / H) * math.pi
    aspect = out_h / out_w
    hfov_from_height = 2 * math.atan(math.tan(ang_h / 2) / aspect)
    hfov = max(ang_w, hfov_from_height)
    return math.degrees(hfov)


# -------------- gaze metrics helpers --------------
def bbox_to_spherical(bbox, W, H):
    x1, y1, x2, y2 = bbox
    lam1, phi1 = _lonlat_from_px(x1, y1, W, H)
    lam2, phi2 = _lonlat_from_px(x2, y2, W, H)
    lam_min, lam_max = sorted([lam1, lam2])
    phi_min, phi_max = sorted([phi1, phi2])
    return lam_min, lam_max, phi_min, phi_max


def gaze_hits_bbox(gaze_vec, bbox, W, H, tol_deg=5.0):
    # gaze vector → lon/lat
    Dx, Dy, Dz = gaze_vec
    lam_g = math.atan2(Dz, Dx)
    phi_g = math.asin(np.clip(Dy, -1.0, 1.0))
    
    # bbox → spherical bounds
    lam_min, lam_max, phi_min, phi_max = bbox_to_spherical(bbox, W, H)
    
    # add tolerance
    tol = math.radians(tol_deg)
    return (lam_min - tol <= lam_g <= lam_max + tol and
            phi_min - tol <= phi_g <= phi_max + tol)


def compute_gaze_matrix(detections, head_poses, W, H):
    """
    detections: list of YOLO outputs (each has bbox, center, conf)
    head_poses: list of (f_vec, ...) for each detection
    Returns NxN matrix: G[i][j] = 1 if i looks at j
    """
    N = len(detections)
    G = np.zeros((N, N), dtype=int)

    for i, (det_i, pose_i) in enumerate(zip(detections, head_poses)):
        if pose_i is None:
            continue
        f_vec = pose_i  # forward vector from PnP
        for j, det_j in enumerate(detections):
            if i == j:
                continue  # skip self
            if gaze_hits_bbox(f_vec, det_j["bbox"], W, H):
                G[i, j] = 1
    return G

# ---------- detection ----------
def load_yolo(model_path):
    return YOLO(model_path)

def detect_persons(bgr, model, conf=0.35, imgsz=960):
    res = model.predict(bgr, verbose=False, imgsz=imgsz, conf=conf, classes=[0])[0]
    out = []
    if res.boxes is None:
        return out
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    for (x1, y1, x2, y2), c in zip(xyxy, confs):
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        out.append({"bbox": [int(x1), int(y1), int(x2), int(y2)], "center": [u, v], "conf": float(c)})
    return out

def select_top3_by_size(detections):
    # Sort by bbox area (descending)
    dets_sorted = sorted(detections,
                         key=lambda d: (d["bbox"][2]-d["bbox"][0]) * (d["bbox"][3]-d["bbox"][1]),
                         reverse=True)
    return dets_sorted[:3]


# ---------- viz helpers ----------

def draw_gaze_matrix_arrows(frame, detections, G, color=(255,0,0)):
    """
    frame: equirect image (BGR)
    detections: list of detections with 'center' (u,v)
    G: NxN gaze matrix (int or float)
    """
    for i, d_i in enumerate(detections):
        u_i, v_i = map(int, d_i["center"])
        for j, d_j in enumerate(detections):
            if i == j or G[i, j] <= 0.5:  # skip self or weak gaze
                continue
            u_j, v_j = map(int, d_j["center"])
            cv2.arrowedLine(frame, (u_i, v_i), (u_j, v_j),
                COLORS[i % len(COLORS)], 2, tipLength=0.15)
            




# ---------- main ----------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(SAVE_DIR, f"headpose_{session_stamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_iso", "frame_idx", "person_id", "yaw", "pitch", "roll", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "center_u", "center_v", "fov_deg", "crop_w", "crop_h", "image_path"])

    # Load models
    yolo = load_yolo(YOLO_MODEL_PATH)

    # Load camera or video
    if USE_CAMERA:
        cap, backend_label, source_label = find_obs_or_insta_camera(
            width=CAM_WIDTH, height=CAM_HEIGHT, prefer_equirect=True)
        if cap is None:
            raise RuntimeError("No usable camera found.")
        print(f"[INFO] Using camera: {source_label} via {backend_label}")
    else:
        cap = cv2.VideoCapture(VIDEO_FILE_PATH)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {VIDEO_FILE_PATH}")
        print(f"[INFO] Using video file: {VIDEO_FILE_PATH}")


    last_yolo_t = last_pose_t = 0.0
    current_dets, latest_head_poses = [], []
    frame_idx = 0

    # Run real-time capture or run from video file
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        # print("Aspect ratio:", W / H)
        frame_idx += 1
        t_now = time.time()

        if (t_now - last_yolo_t) >= YOLO_INTERVAL_SEC:
            current_dets = detect_persons(frame, yolo, conf=YOLO_CONF_THRESHOLD, imgsz=YOLO_IMGSZ)
            current_dets = select_top3_by_size(current_dets)
            for pid, d in enumerate(current_dets, start=1):
                d["id"] = f"P{pid}"
                d["color"] = COLORS[(pid-1) % len(COLORS)]
            last_yolo_t = t_now

        equi_with_pose = frame.copy()
        if SHOW_PERSON_BBOX:
            for d in current_dets:
                x1, y1, x2, y2 = d["bbox"]
                color = d["color"]
                cv2.rectangle(equi_with_pose, (x1, y1), (x2, y2), color, 2)
                cv2.putText(equi_with_pose, d["id"], (x1, max(20, y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if (t_now - last_pose_t) >= POSE_INTERVAL_SEC and current_dets:
            for pid, d in enumerate(current_dets, start=1):
                u, v = d["center"]
                hfov_deg = fov_from_bbox_exact(d["bbox"], (u, v), W, H, GNOMONIC_OUT_W, GNOMONIC_OUT_H)
                crop = equirect_to_gnomonic(frame, center_px=(u, v), fov_deg=hfov_deg, out_w=GNOMONIC_OUT_W, out_h=GNOMONIC_OUT_H, roll_deg=GNOMONIC_ROLL_DEG)
                pose = estimate_head_pose_with_visuals(crop)

                if pose:
                    pitch, yaw, roll, head_box, nose_tip, nose_end, rvec = pose
                    r_vec, u_vec, f_vec = _basis_from_forward(_dir_from_lonlat(*_lonlat_from_px(u, v, W, H)))
                    face_cx = (head_box[0] + head_box[2]) / 2
                    face_cy = (head_box[1] + head_box[3]) / 2
                    base_px = dir_to_equirect_px(crop_px_to_global_dir(face_cx, face_cy, GNOMONIC_OUT_W, GNOMONIC_OUT_H, hfov_deg, r_vec, u_vec, f_vec), W, H)
                    end_px = dir_to_equirect_px(_unit(f_vec), W, H)
                    if SHOW_FACE_BBOX:
                        corners_crop = [(head_box[0], head_box[1]), (head_box[2], head_box[1]), (head_box[2], head_box[3]), (head_box[0], head_box[3])]
                        corners_eq = [dir_to_equirect_px(crop_px_to_global_dir(cx, cy, GNOMONIC_OUT_W, GNOMONIC_OUT_H, hfov_deg, r_vec, u_vec, f_vec), W, H) for cx, cy in corners_crop]
                        cv2.polylines(equi_with_pose, [np.array(corners_eq, np.int32)], isClosed=True, color=color, thickness=2)
                        cv2.arrowedLine(equi_with_pose, base_px, end_px, color, 2, tipLength=0.25)
                    
                    if SAVE_COMPARISONS:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(os.path.join(SAVE_DIR, f"gnomonic_{pid}_{ts}.jpg"), crop)
                        # cv2.imwrite(os.path.join(SAVE_DIR, f"equi_{pid}_{ts}.jpg"), equi_with_pose)
                        cv2.imwrite(os.path.join(SAVE_DIR, f"equi_{pid}_{ts}.jpg"), frame)

                all_head_poses = []
                for pid, d in enumerate(current_dets, start=1):
                    # ... your crop + head pose code ...
                    if pose:
                        pitch, yaw, roll, head_box, nose_tip, nose_end, rvec = pose
                        f_vec = _unit(f_vec)  # already from your projection step
                        all_head_poses.append(f_vec)
                    else:
                        all_head_poses.append(None)

                # Now compute NxN gaze matrix
                G = compute_gaze_matrix(current_dets, all_head_poses, W, H)
                print("Gaze matrix:\n", G)

                # Save current frame's gaze matrix
                GAZE_HISTORY.append(G)

                # Compute average over history
                if len(GAZE_HISTORY) > 0:
                    avg_G = np.mean(GAZE_HISTORY, axis=0)   # elementwise average
                else:
                    avg_G = None

                for i in range(len(current_dets)):
                    row = [datetime.now().isoformat(), frame_idx, i+1]
                    row += list(G[i])  # instantaneous matrix row
                    if avg_G is not None:
                        row += list(np.round(avg_G[i], 3))  # averaged (rounded)
                    CSV_BUFFER.append(row)

                y_offset = 30

                for i, d_i in enumerate(current_dets):
                    for j, d_j in enumerate(current_dets):
                        if i != j and G[i, j] == 1:
                            txt = f"{d_i['id']} → {d_j['id']}"
                            cv2.putText(equi_with_pose, txt, (50, y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, d_i["color"], 2)
                            y_offset += 25

                            # highlight target box with gazer’s color
                            x1, y1, x2, y2 = d_j["bbox"]
                            cv2.rectangle(equi_with_pose, (x1, y1), (x2, y2), d_i["color"], 2)

                if frame_idx % CSV_FLUSH_INTERVAL == 0 and CSV_BUFFER:
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(CSV_BUFFER)
                    CSV_BUFFER.clear()
                    print(f"[INFO] Flushed {CSV_FLUSH_INTERVAL} rows to CSV.")

                # Draw gaze arrows
                if SHOW_EQUIRECT_WINDOW and len(current_dets) > 1:
                    draw_gaze_matrix_arrows(equi_with_pose, current_dets, G)

            last_pose_t = t_now


        if SHOW_EQUIRECT_WINDOW:
            disp_w = min(EQUIRECT_DISPLAY_MAX_W, W)
            disp_h = int(disp_w / (W / float(H)))
            equi_disp = cv2.resize(equi_with_pose, (disp_w, disp_h))
            cv2.imshow("Equirect Preview", equi_disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    
    # Final flush
    if CSV_BUFFER:
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(CSV_BUFFER)
        print(f"[INFO] Final flush: wrote {len(CSV_BUFFER)} rows.")

    
    cap.release()
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


if __name__ == "__main__":
    import cProfile, pstats, io
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(30)   # show top 30 functions
    print(s.getvalue())
