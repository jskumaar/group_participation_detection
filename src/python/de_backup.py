#!/usr/bin/env python3
import cv2, numpy as np, math, os, time, csv
from datetime import datetime
from ultralytics import YOLO
import mediapipe as mp
import sys

# =========================
# CONFIG
# =========================
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
SHOW_PREVIEW        = False
SHOW_EQUIRECT_WINDOW = True
EQUIRECT_DISPLAY_MAX_W = 1280

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
    "OBS-Camera",  # just in case
]

def _open_dshow_by_name(name, width=None, height=None):
    """Windows/DirectShow: open device by name string like 'video=OBS Virtual Camera'."""
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
    """Grab a few frames; return last good frame (or None)."""
    frame = None
    for _ in range(warmup):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
        else:
            time.sleep(0.02)
    return frame

def _is_equirectangular(frame, ratio_tol=0.08):
    """Heuristic: equirect frames are ~2:1 (width:height)."""
    h, w = frame.shape[:2]
    ratio = w / float(h)
    return abs(ratio - 2.0) <= ratio_tol

def find_obs_or_insta_camera(
    preferred_names=PREFERRED_NAMES,
    width=None, height=None,
    prefer_equirect=True,
):
    """
    Returns (cap, backend_label, source_label).
    Tries, in order:
      1) Windows/DirectShow: open by name (OBS/Insta) if available
      2) Scan indices on common backends; pick the first with a valid frame
         and prefer ~2:1 aspect if prefer_equirect=True
    """
    system = sys.platform
    # 1) Windows: try DirectShow by "video=<name>"
    if system.startswith("win"):
        for name in preferred_names:
            cap, frame = _open_dshow_by_name(name, width, height)
            if cap is not None:
                return cap, "CAP_DSHOW(name)", name

    # 2) Fallback: scan indices/backends
    backends = []
    if system.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    elif system == "darwin":
        backends = [cv2.CAP_AVFOUNDATION]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    candidates = []
    for backend in backends:
        for idx in range(6):  # scan a handful of indices
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

    # Prefer equirect (2:1) if requested, else first valid
    if prefer_equirect:
        for cap, backend, idx, w, h, is_eq in candidates:
            if is_eq:
                return cap, f"backend={backend}", f"idx={idx} ({w}x{h})"
    cap, backend, idx, w, h, _ = candidates[0]
    return cap, f"backend={backend}", f"idx={idx} ({w}x{h})"



def compute_crop_basis(center_px, yaw0_deg, roll_deg, W, H):
    """Return r,u,f (unit vectors in global 360 frame) for a crop centered at center_px."""
    lam, phi = _lonlat_from_px(center_px[0], center_px[1], W, H)
    lam += math.radians(yaw0_deg)
    fwd = _dir_from_lonlat(lam, phi)
    r, u, f = _basis_from_forward(fwd)
    r, u = _apply_roll(r, u, math.radians(roll_deg))
    return r, u, f  # each is (3,)

def slerp_dir(a, b, max_deg):
    """Short arc slerp from unit a->b by at most max_deg (degrees)."""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    omega = math.acos(dot)
    if omega < 1e-6:
        return b
    t = min(math.radians(max_deg) / omega, 1.0)
    s1 = math.sin((1 - t) * omega) / math.sin(omega)
    s2 = math.sin(t * omega) / math.sin(omega)
    return s1 * a + s2 * b

def dir_to_equirect_px(D, W, H):
    """Unit direction vector -> (u,v) in equirect pixels."""
    Dx, Dy, Dz = D
    lam = math.atan2(Dz, Dx)            # [-pi, pi]
    phi = math.asin(np.clip(Dy, -1.0, 1.0))  # [-pi/2, pi/2]
    u = (lam + math.pi) / (2 * math.pi) * W
    v = (math.pi/2 - phi) / math.pi * H
    return int(round(u)), int(round(v))


def get_camera_matrix(w, h):
    focal_length = w
    center = (w / 2, h / 2)
    return np.array([[focal_length, 0, center[0]],
                     [0, focal_length, center[1]],
                     [0, 0, 1]], dtype="double")




def estimate_head_pose_with_visuals(bgr):
    """
    Returns (pitch, yaw, roll, head_box(x1,y1,x2,y2), nose_tip_2d, nose_end_2d) or None.
    """
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

    success, rvec, tvec = cv2.solvePnP(
        FACE_3D_MODEL, face_2d, cam_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = [math.degrees(a) for a in angles]

    axis_len = 100.0
    nose_end_3d = np.array([(0.0, 0.0, axis_len)], dtype=np.float64)
    nose_end_2d, _ = cv2.projectPoints(nose_end_3d, rvec, tvec, cam_matrix, dist)

    nose_tip_2d = (int(face_2d[0, 0]), int(face_2d[0, 1]))  # nose tip is first
    nose_end_2d = tuple(map(int, nose_end_2d[0].ravel()))
    return pitch, yaw, roll, head_box, nose_tip_2d, nose_end_2d, rvec

# =========================
# Equirect helpers (gnomonic)
# =========================
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
    r = np.cross(u, f)  # right-handed
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

def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / (n + 1e-9)

def _ang_deg(a, b):
    a = _unit(a); b = _unit(b)
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))

def _dir_from_px(u, v, W, H):
    lam, phi = _lonlat_from_px(u, v, W, H)
    return _dir_from_lonlat(lam, phi)

def _dir_from_lonlat_deg(lon_deg, lat_deg):
    return _dir_from_lonlat(math.radians(lon_deg), math.radians(lat_deg))


# =========================
# YOLO helpers
# =========================
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
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        out.append({"bbox": [x1, y1, x2, y2], "center": [u, v], "conf": float(c)})
    return out

def assign_gaze_targets(latest_head_poses, detections, W, H):
    results = []
    person_dirs = []
    for d in detections:
        u, v = d["center"]
        person_dirs.append(_unit(_dir_from_px(u, v, W, H)))
    robot_dir = _unit(_dir_from_lonlat_deg(*ROBOT_LONLAT)) if USE_ROBOT else None
    for pid, hp in enumerate(latest_head_poses, start=1):
        gaze = _unit(hp["v_global"])
        best = ("outside", None, 180.0)
        for qid, tgt_dir in enumerate(person_dirs, start=1):
            if qid == pid:
                continue
            if np.dot(gaze, tgt_dir) <= GAZE_FRONT_COS:
                continue
            ang = _ang_deg(gaze, tgt_dir)
            if ang < best[2]:
                best = ("person", qid, ang)
        if robot_dir is not None and np.dot(gaze, robot_dir) > GAZE_FRONT_COS:
            ang = _ang_deg(gaze, robot_dir)
            if ang < best[2]:
                best = ("robot", 0, ang)
        if gaze[1] < -0.35 and best[2] > 20.0:
            best = ("down", None, 0.0)
        if best[0] in ("person", "robot") and best[2] > GAZE_MAX_DEG:
            best = ("outside", None, best[2])
        results.append({
            "from_id": pid,
            "target_type": best[0],
            "target_id": best[1],
            "angle_deg": best[2]
        })
    return results

# =========================
# Main loop
# =========================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(SAVE_DIR, f"headpose_{session_stamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_iso", "frame_idx", "person_id",
                         "yaw", "pitch", "roll",
                         "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                         "center_u", "center_v", "fov_deg", "crop_w", "crop_h",
                         "image_path", "target_type", "target_id", "target_angle_deg"])

    yolo = load_yolo(YOLO_MODEL_PATH)
    cap, backend_label, source_label = find_obs_or_insta_camera(
        width=CAM_WIDTH, height=CAM_HEIGHT, prefer_equirect=True
    )
    if cap is None:
        raise RuntimeError("No usable camera found.")
    print(f"[INFO] Using camera: {source_label} via {backend_label}")

    last_yolo_t = 0.0
    last_pose_t = 0.0
    current_dets = []
    latest_head_poses = []
    gaze_links = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_now = time.time()
        H, W = frame.shape[:2]
        frame_idx += 1

        # Run YOLO periodically
        if (t_now - last_yolo_t) >= YOLO_INTERVAL_SEC:
            current_dets = detect_persons(frame, yolo, conf=YOLO_CONF_THRESHOLD, imgsz=YOLO_IMGSZ)
            last_yolo_t = t_now

        # Run head pose periodically
        if (t_now - last_pose_t) >= POSE_INTERVAL_SEC and current_dets:
            latest_head_poses = []
            pose_rows = []
            for pid, d in enumerate(current_dets, start=1):
                x1, y1, x2, y2 = map(int, d["bbox"])
                u, v = d["center"]
                if PROJECTION_MODE.lower() == "gnomonic":
                    hfov_deg = fov_from_bbox_exact(d["bbox"], (u, v), W, H, GNOMONIC_OUT_W, GNOMONIC_OUT_H)
                    crop = equirect_to_gnomonic(frame, center_px=(u, v), fov_deg=hfov_deg,
                                                out_w=GNOMONIC_OUT_W, out_h=GNOMONIC_OUT_H,
                                                roll_deg=GNOMONIC_ROLL_DEG)
                    fov_used = hfov_deg
                else:
                    crop = frame[y1:y2, x1:x2].copy()
                    fov_used = float("nan")
                pose = estimate_head_pose_with_visuals(crop)
                yaw = pitch = roll = ""
                if pose:
                    pitch, yaw, roll, head_box, nose_tip, nose_end, rvec = pose
                    roll_used = GNOMONIC_ROLL_DEG + CALIB_ROLL_DEG if PROJECTION_MODE=='gnomonic' else CALIB_ROLL_DEG
                    r_vec, u_vec, f_vec = compute_crop_basis((u, v), CALIB_YAW_DEG, roll_used, W, H)
                    R_cam, _ = cv2.Rodrigues(rvec)
                    v_cam = (R_cam @ np.array([0.0, 0.0, 1.0])).astype(np.float64)
                    v_global = (r_vec * v_cam[0]) + (u_vec * v_cam[1]) + (f_vec * v_cam[2])
                    latest_head_poses.append({
                        "center_px": (int(u), int(v)),
                        "bbox": d["bbox"],
                        "f_vec": f_vec,
                        "v_global": v_global,
                        "yaw": float(yaw),
                        "pitch": float(pitch)
                    })
                ts_iso = datetime.now().isoformat(timespec="milliseconds")
                bx1, by1, bx2, by2 = d["bbox"]
                pose_rows.append([ts_iso, frame_idx, pid,
                                  yaw, pitch, roll,
                                  bx1, by1, bx2, by2,
                                  u, v, fov_used, crop.shape[1], crop.shape[0],
                                  "", "", "", ""])
            gaze_links = assign_gaze_targets(latest_head_poses, current_dets, W, H) if latest_head_poses else []
            link_by_id = {l["from_id"]: l for l in gaze_links}
            for row in pose_rows:
                pid = row[2]
                lnk = link_by_id.get(pid, {"target_type":"", "target_id":"", "angle_deg":""})
                row[-3:] = [lnk["target_type"], lnk["target_id"], lnk["angle_deg"]]
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(pose_rows)
            last_pose_t = t_now

        # Always draw latest results
        equi_with_pose = frame.copy()
        for item in latest_head_poses:
            (cx, cy) = item["center_px"]
            x1, y1, x2, y2 = map(int, item["bbox"])
            cv2.rectangle(equi_with_pose, (x1, y1), (x2, y2), (0,255,0), 2)
            f_vec = item["f_vec"]
            v_g = item["v_global"]
            end_dir = slerp_dir(f_vec, v_g, max_deg=12.0)
            end_px  = dir_to_equirect_px(end_dir / np.linalg.norm(end_dir), W, H)
            cv2.arrowedLine(equi_with_pose, (cx, cy), end_px, (0, 0, 255), 2, tipLength=0.2)
            cv2.putText(equi_with_pose, f"Y:{item['yaw']:.0f} P:{item['pitch']:.0f}",
                        (cx+8, cy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

        for link in gaze_links:
            try:
                src = latest_head_poses[link["from_id"]-1]
            except IndexError:
                continue
            (cx, cy) = src["center_px"]
            if link["target_type"] == "person":
                tgt_det = current_dets[link["target_id"]-1]
                tu, tv = map(int, tgt_det["center"])
                cv2.line(equi_with_pose, (cx, cy), (tu, tv), (255, 0, 0), 2)
            elif link["target_type"] == "robot" and USE_ROBOT:
                end_px = dir_to_equirect_px(_unit(_dir_from_lonlat_deg(*ROBOT_LONLAT)), W, H)
                cv2.line(equi_with_pose, (cx, cy), end_px, (0, 165, 255), 2)

        if SHOW_EQUIRECT_WINDOW:
            disp_w = min(EQUIRECT_DISPLAY_MAX_W, W)
            disp_h = int(disp_w / (W / float(H)))
            equi_disp = cv2.resize(equi_with_pose, (disp_w, disp_h))
            cv2.imshow("Equirect Preview", equi_disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()