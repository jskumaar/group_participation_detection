#!/usr/bin/env python3
import cv2, numpy as np, math, os, time, csv
from datetime import datetime
from ultralytics import YOLO
import mediapipe as mp

import sys, subprocess, shlex

# =========================
# CONFIG (edit these)
# =========================
# Intervals
YOLO_INTERVAL_SEC   = 10.0      # seconds between person detections
POSE_INTERVAL_SEC   = 1.0      # seconds between head pose estimations

# Camera (OBS Virtual Camera). Use your equirectangular output size here.
CAMERA_INDEX        = 1
CAM_WIDTH           = 2880     # e.g., 5760 for 6K equirect
CAM_HEIGHT          = 1440     # e.g., 2880 for 6K equirect (2:1)

# YOLO
YOLO_MODEL_PATH     = "yolov8n.pt"
YOLO_CONF_THRESHOLD = 0.35
YOLO_IMGSZ          = 960

# Projection mode for pose crops: "gnomonic" or "none"
PROJECTION_MODE     = "gnomonic"
GNOMONIC_OUT_W      = 512
GNOMONIC_OUT_H      = 512
GNOMONIC_ROLL_DEG   = 180.0    # flip if your projection looks upside-down

# Output & UI
SAVE_DIR            = "live_out"
SHOW_PREVIEW        = False     # show preview window with YOLO boxes + quick pose text
SHOW_EQUIRECT_WINDOW = True   # new window for full equirect frame
EQUIRECT_DISPLAY_MAX_W = 1280 # scale-down width for display (keeps 2:1 aspect)


# =========================
# FaceMesh + PnP setup
# =========================
FACE_3D_IDXS = [1, 152, 33, 263, 61, 291]
FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -63.6, -12.5),    # Chin
    (-43.3, 32.7, -26.0),   # Left eye left corner
    (43.3, 32.7, -26.0),    # Right eye right corner
    (-28.9, -28.9, -24.1),  # Left Mouth corner
    (28.9, -28.9, -24.1)    # Right mouth corner
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
    return pitch, yaw, roll, head_box, nose_tip_2d, nose_end_2d

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

# =========================
# Live loop
# =========================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # CSV log per session
    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(SAVE_DIR, f"headpose_{session_stamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_iso", "frame_idx", "person_id",
                         "yaw", "pitch", "roll",
                         "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                         "center_u", "center_v", "fov_deg", "crop_w", "crop_h",
                         "image_path"])

    # Models
    yolo = load_yolo(YOLO_MODEL_PATH)

    # Camera
    # cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows
    
    # OPEN CAMERA (tries OBS/Insta by name first; else scans indices)
    cap, backend_label, source_label = find_obs_or_insta_camera(
        width=CAM_WIDTH, height=CAM_HEIGHT, prefer_equirect=True
    )
    if cap is None:
        raise RuntimeError("No usable camera. Is OBS Virtual Camera started?")

    print(f"[INFO] Using camera: {source_label} via {backend_label}")

    # (Optional) verify frames aren't blank (scene hidden in OBS)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Camera opened, but no frames. Check OBS scene/sources.")
    if frame.mean() < 1.0:
        print("[WARN] Frame is black. Your OBS scene may be empty or sources hidden.")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try a different CAMERA_INDEX (OBS is often 0/1).")

    last_yolo_t = 0.0
    last_pose_t = 0.0
    current_dets = []  # reused between YOLO runs
    frame_idx = 0

    print("[INFO] Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Camera read failed.")
            break

        t_now = time.time()
        H, W = frame.shape[:2]
        frame_idx += 1

        # Run YOLO at interval
        if (t_now - last_yolo_t) >= YOLO_INTERVAL_SEC:
            current_dets = detect_persons(frame, yolo, conf=YOLO_CONF_THRESHOLD, imgsz=YOLO_IMGSZ)
            last_yolo_t = t_now

        # Draw person boxes on preview
        preview = frame.copy()
        for i, d in enumerate(current_dets):
            x1, y1, x2, y2 = map(int, d["bbox"])
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(preview, f"id{i} {d['conf']:.2f}", (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

        # Run head pose at interval for all current detections
        if (t_now - last_pose_t) >= POSE_INTERVAL_SEC:
            pose_rows = []
            for pid, d in enumerate(current_dets, start=1):
                x1, y1, x2, y2 = map(int, d["bbox"])
                u, v = d["center"]

                # Obtain a crop for pose
                if PROJECTION_MODE.lower() == "gnomonic":
                    hfov_deg = fov_from_bbox_exact(d["bbox"], (u, v), W, H, GNOMONIC_OUT_W, GNOMONIC_OUT_H)
                    crop = equirect_to_gnomonic(
                        frame, center_px=(u, v), fov_deg=hfov_deg,
                        out_w=GNOMONIC_OUT_W, out_h=GNOMONIC_OUT_H, roll_deg=GNOMONIC_ROLL_DEG
                    )
                    fov_used = hfov_deg
                else:
                    crop = frame[y1:y2, x1:x2].copy()
                    fov_used = float("nan")

                # Head pose on crop
                pose = estimate_head_pose_with_visuals(crop)
                image_path_saved = ""
                yaw = pitch = roll = ""
                if pose:
                    pitch, yaw, roll, head_box, nose_tip, nose_end = pose

                    # Overlay (box + arrow + text) on crop and save
                    annotated = crop.copy()
                    hx1, hy1, hx2, hy2 = map(int, head_box)
                    cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
                    cv2.arrowedLine(annotated, nose_tip, nose_end, (0, 0, 255), 2, tipLength=0.2)
                    cv2.putText(annotated, f"Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}",
                                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms
                    stem = f"pose_p{pid}_f{frame_idx}_{ts}"
                    image_path_saved = os.path.join(SAVE_DIR, f"{stem}.jpg")
                    cv2.imwrite(image_path_saved, annotated)

                    # Quick text on preview near bbox
                    cv2.putText(preview, f"Y:{yaw:.0f} P:{pitch:.0f} R:{roll:.0f}",
                                (x1, y1-8 if y1-8 > 12 else y1+15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

                # Log row (even if pose failed, we log attempt)
                ts_iso = datetime.now().isoformat(timespec="milliseconds")
                bx1, by1, bx2, by2 = d["bbox"]
                pose_rows.append([ts_iso, frame_idx, pid,
                                  yaw, pitch, roll,
                                  bx1, by1, bx2, by2,
                                  u, v, fov_used, crop.shape[1], crop.shape[0],
                                  image_path_saved])

                # ---------- (Comparison code DISABLED; re-enable if needed) ----------
                # if PROJECTION_MODE.lower() == "gnomonic":
                #     unproj_crop = frame[y1:y2, x1:x2]
                #     unproj_resized = cv2.resize(unproj_crop, (crop.shape[1], crop.shape[0]))
                #     side_by_side = np.hstack([unproj_resized, annotated if pose else crop])
                #     overlay_img = cv2.addWeighted(unproj_resized, 0.5,
                #                                   annotated if pose else crop, 0.5, 0)
                #     cv2.imwrite(os.path.join(SAVE_DIR, f"{stem}_compare.jpg"), side_by_side)
                #     cv2.imwrite(os.path.join(SAVE_DIR, f"{stem}_overlay.jpg"), overlay_img)
                # --------------------------------------------------------------------

            if pose_rows:
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(pose_rows)

            last_pose_t = t_now

        if SHOW_PREVIEW:
            cv2.imshow("Live (OBS Virtual Cam)", preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if SHOW_EQUIRECT_WINDOW:
            # Show the full raw equirect frame (scaled for screen)
            disp_w = min(EQUIRECT_DISPLAY_MAX_W, W)
            disp_h = int(disp_w / (W / float(H)))  # preserve aspect (≈2:1)
            equi_disp = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

            cv2.imshow("Equirect Preview (raw)", equi_disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Pose log saved to: {csv_path}")


if __name__ == "__main__":
    main()
