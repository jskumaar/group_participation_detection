# #!/usr/bin/env python3
# import os, time, math, csv
# from datetime import datetime
# from collections import deque

# import cv2
# import numpy as np
# from ultralytics import YOLO
# import mediapipe as mp

# # =========================
# # CONFIG
# # =========================
# USE_CAMERA = False
# # VIDEO_FILE_PATH = r"G:\My Drive\Research\(Shared) Project_mixed_group_interaction\Pilot_data_collection\Session_1\Pilot_1_360_hd_interaction.mp4"
# VIDEO_FILE_PATH = r"G:\My Drive\Research\(Shared) Project_mixed_group_interaction\Pilot_data_collection\Session_2\pilot_2_360_hd_interaction_corrected.mp4"
# # VIDEO_FILE_PATH = r"C:\Users\sures\Desktop\children_pilot_2_360_corrected_with_audio.mp4"


# STEP_MODE = True  # True = space to step frame-by-frame, False = continuous playback

# SAVE_DIR = "live_out"
# SHOW_EQUIRECT_WINDOW    = True
# EQUIRECT_DISPLAY_MAX_W  = 1280

# # Drawing toggles
# SHOW_PERSON_BBOX = True
# SHOW_FACE_BBOX   = True   # keeps the projected face bbox poly; arrows are simplified

# # Camera (if using live)
# CAMERA_INDEX  = 0
# CAM_WIDTH     = 2880
# CAM_HEIGHT    = 1440

# # YOLO
# YOLO_MODEL_PATH     = "yolov8n.pt"
# YOLO_CONF_THRESHOLD = 0.35
# YOLO_IMGSZ          = 960

# # Gnomonic (local synthetic pinhole)
# GNOMONIC_OUT_W    = 512
# GNOMONIC_OUT_H    = 512
# GNOMONIC_HFOV_DEG = 80.0     # horizontal FoV for the gnomonic crop
# GNOMONIC_ROLL_DEG = 180.0

# # Social-gaze params
# HEAD_WIDTH_M    = 0.135      # average (tune for age group)
# GAZE_THRESH_DEG = 8.0        # angle tolerance for "looks_at"

# # Person colors
# COLORS = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,128,0), (255,0,255)]

# # Intervals
# YOLO_INTERVAL_SEC = 10.0     # run person detection every N seconds
# POSE_INTERVAL_SEC = 0.1      # run head pose / gaze every N seconds

# # History smoothing (optional)
# GAZE_HISTORY = deque(maxlen=30)

# os.makedirs(SAVE_DIR, exist_ok=True)

# # =========================
# # Mediapipe FaceMesh (PnP)
# # =========================
# mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
#     max_num_faces=1, refine_landmarks=True,
#     min_detection_confidence=0.5, min_tracking_confidence=0.5
# )

# # Subset of 3D model points (correspond to FaceMesh IDs below)
# FACE_3D_IDXS = [1, 152, 33, 263, 61, 291]
# FACE_3D_MODEL = np.array([
#     (0.0, 0.0, 0.0),        # nose tip
#     (0.0, -63.6, -12.5),    # chin
#     (-43.3, 32.7, -26.0),   # left eye corner
#     (43.3, 32.7, -26.0),    # right eye corner
#     (-28.9, -28.9, -24.1),  # left mouth corner
#     (28.9, -28.9, -24.1)    # right mouth corner
# ], dtype=np.float64)

# # =========================
# # Geometry helpers
# # =========================
# def _unit(v):
#     v = np.asarray(v, dtype=np.float64)
#     n = np.linalg.norm(v)
#     return v / (n + 1e-9)

# def px_to_dir_equirect(u, v, W, H):
#     lon = (u / float(W)) * 2.0 * math.pi - math.pi
#     lat = math.pi/2.0 - (v / float(H)) * math.pi
#     x = math.cos(lat) * math.cos(lon)
#     y = math.sin(lat)
#     z = math.cos(lat) * math.sin(lon)
#     return np.array([x, y, z], dtype=np.float64)

# def dir_to_equirect_px(D, W, H):
#     Dx, Dy, Dz = D
#     lam = math.atan2(Dz, Dx)
#     phi = math.asin(np.clip(Dy, -1.0, 1.0))
#     u = (lam + math.pi) / (2 * math.pi) * W
#     v = (math.pi/2 - phi) / math.pi * H
#     return int(round(u)), int(round(v))

# def _dir_from_lonlat(lam, phi):
#     c = np.cos(phi)
#     return np.array([c*np.cos(lam), np.sin(phi), c*np.sin(lam)], dtype=np.float64)

# def _lonlat_from_px(u, v, W, H):
#     lam = 2*math.pi*(u/W) - math.pi
#     phi = math.pi/2 - math.pi*(v/H)
#     return lam, phi

# def _basis_from_forward(f):
#     f = f/np.linalg.norm(f)
#     up = np.array([0, 1, 0.0], dtype=np.float64)
#     u = up - f*np.dot(up, f)
#     if np.linalg.norm(u) < 1e-9:
#         up = np.array([1, 0, 0.0], dtype=np.float64)
#         u = up - f*np.dot(up, f)
#     u = u/np.linalg.norm(u)
#     r = np.cross(u, f)
#     r = r/np.linalg.norm(r)
#     return r, u, f

# def _apply_roll(r, u, roll_rad):
#     c, s = math.cos(roll_rad), math.sin(roll_rad)
#     r2 = c*r + s*u
#     u2 = c*u - s*r
#     return r2, u2

# def equirect_to_gnomonic(equi_bgr, *, center_px, fov_deg, out_w, out_h, roll_deg):
#     H, W = equi_bgr.shape[:2]
#     lam, phi = _lonlat_from_px(center_px[0], center_px[1], W, H)
#     fwd = _dir_from_lonlat(lam, phi)
#     r, u, f = _basis_from_forward(fwd)
#     r, u = _apply_roll(r, u, math.radians(roll_deg))

#     hfov = math.radians(fov_deg)
#     vfov = 2*math.atan(math.tan(hfov/2) * (out_h/out_w))
#     xs = np.linspace(-math.tan(hfov/2), math.tan(hfov/2), out_w, dtype=np.float64)
#     ys = np.linspace(-math.tan(vfov/2), math.tan(vfov/2), out_h, dtype=np.float64)
#     X, Y = np.meshgrid(xs, ys); Z = np.ones_like(X)
#     denom = np.sqrt(X*X + Y*Y + Z*Z)
#     lx, ly, lz = X/denom, Y/denom, Z/denom

#     Dx = r[0]*lx + u[0]*ly + f[0]*lz
#     Dy = r[1]*lx + u[1]*ly + f[1]*lz
#     Dz = r[2]*lx + u[2]*ly + f[2]*lz
#     lam = np.arctan2(Dz, Dx)
#     phi = np.arcsin(np.clip(Dy, -1.0, 1.0))
#     mapx = ((lam + math.pi) / (2*math.pi) * W).astype(np.float32)
#     mapy = ((math.pi/2 - phi) / math.pi * H).astype(np.float32)

#     crop = cv2.remap(equi_bgr, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
#     return crop, (r, u, f)

# def crop_px_to_global_dir(px, py, crop_w, crop_h, fov_deg, r_vec, u_vec, f_vec):
#     x_ndc = (px / crop_w - 0.5) * 2
#     y_ndc = (py / crop_h - 0.5) * 2
#     hfov = math.radians(fov_deg)
#     vfov = 2 * math.atan(math.tan(hfov/2) * (crop_h/crop_w))
#     lx = math.tan(hfov/2) * x_ndc
#     ly = math.tan(vfov/2) * y_ndc
#     lz = 1.0
#     dir_cam = np.array([lx, ly, lz], dtype=np.float64)
#     dir_cam /= np.linalg.norm(dir_cam)
#     return _unit(r_vec*dir_cam[0] + u_vec*dir_cam[1] + f_vec*dir_cam[2])

# # =========================
# # Head pose estimation (FaceMesh + PnP)
# # =========================
# def get_camera_matrix(w, h):
#     focal_length = w
#     center = (w / 2, h / 2)
#     return np.array([[focal_length, 0, center[0]],
#                      [0, focal_length, center[1]],
#                      [0, 0, 1]], dtype="double")

# def estimate_head_pose_with_visuals(bgr):
#     h, w = bgr.shape[:2]
#     if h <= 0 or w <= 0:
#         return None
#     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#     results = mp_face_mesh.process(rgb)
#     if not results.multi_face_landmarks:
#         return None
#     lms = results.multi_face_landmarks[0]
#     xs, ys = [], []
#     for lm in lms.landmark:
#         xs.append(int(lm.x * w)); ys.append(int(lm.y * h))
#     head_box = (min(xs), min(ys), max(xs), max(ys))

#     face_2d = np.array([(lms.landmark[i].x * w, lms.landmark[i].y * h) for i in FACE_3D_IDXS], dtype=np.float64)
#     cam_matrix = get_camera_matrix(w, h)
#     dist = np.zeros((4, 1), dtype=np.float64)
#     success, rvec, tvec = cv2.solvePnP(FACE_3D_MODEL, face_2d, cam_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE)
#     if not success:
#         return None
#     return head_box, rvec, lms

# # =========================
# # YOLO person detection
# # =========================
# def load_yolo(model_path):
#     return YOLO(model_path)

# def detect_persons(bgr, model, conf=0.35, imgsz=960):
#     res = model.predict(bgr, verbose=False, imgsz=imgsz, conf=conf, classes=[0])[0]
#     out = []
#     if res.boxes is None:
#         return out
#     xyxy = res.boxes.xyxy.cpu().numpy()
#     confs = res.boxes.conf.cpu().numpy()
#     for (x1, y1, x2, y2), c in zip(xyxy, confs):
#         u = (x1 + x2) / 2.0
#         v = (y1 + y2) / 2.0
#         out.append({"bbox": [int(x1), int(y1), int(x2), int(y2)],
#                     "center": [u, v], "conf": float(c)})
#     return out

# def select_top3_by_size(detections):
#     dets_sorted = sorted(detections,
#                          key=lambda d: (d["bbox"][2]-d["bbox"][0]) * (d["bbox"][3]-d["bbox"][1]),
#                          reverse=True)
#     return dets_sorted[:3]

# # =========================
# # 3D look-at geometry
# # =========================
# def estimate_distance_from_headsize(head_bbox, fx, head_width_m=HEAD_WIDTH_M):
#     w_px = max(head_bbox[2] - head_bbox[0], 1e-6)
#     return (fx * head_width_m) / w_px

# def person_position_from_equi(u, v, W, H, r):
#     D = px_to_dir_equirect(u, v, W, H)
#     return r * (D / np.linalg.norm(D))

# def looks_at(P_i, G_i, P_j, deg_thresh=GAZE_THRESH_DEG):
#     T = P_j - P_i
#     nT = np.linalg.norm(T)
#     if nT < 1e-6:
#         return False
#     cosang = np.dot(G_i, T / nT)
#     return cosang >= math.cos(math.radians(deg_thresh))

# def compute_lookat_matrix(P_list, G_list, deg_thresh=GAZE_THRESH_DEG):
#     N = len(P_list)
#     Gmat = np.zeros((N, N), dtype=int)
#     for i in range(N):
#         for j in range(N):
#             if i != j and looks_at(P_list[i], G_list[i], P_list[j], deg_thresh):
#                 Gmat[i, j] = 1
#     return Gmat

# # Optional: build gaze from Euler if needed (not used by default)
# def gaze_from_euler(yaw_deg, pitch_deg, roll_deg=0.0):
#     yaw = math.radians(yaw_deg)
#     pitch = math.radians(pitch_deg)
#     gx = math.cos(pitch) * math.cos(yaw)
#     gy = math.sin(pitch)
#     gz = math.cos(pitch) * math.sin(yaw)
#     return np.array([gx, gy, gz], dtype=float) / np.linalg.norm([gx, gy, gz])

# # =========================
# # Visualization helpers
# # =========================
# def project_position_to_equirect_px(P, W, H):
#     if np.linalg.norm(P) < 1e-9:
#         return None
#     D = _unit(P)
#     return dir_to_equirect_px(D, W, H)

# def draw_people_and_ids(frame, people_px, colors, ids):
#     for idx, p in enumerate(people_px):
#         if p is None:
#             continue
#         u, v = p
#         col = colors[idx % len(colors)]
#         cv2.circle(frame, (u, v), 8, col, -1)
#         cv2.putText(frame, ids[idx], (u+10, max(20, v-10)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

# def draw_lookat_arrows(frame, people_px, Gmat, colors):
#     N = len(people_px)
#     for i in range(N):
#         pi = people_px[i]
#         if pi is None:
#             continue
#         for j in range(N):
#             if i == j or Gmat[i, j] == 0:
#                 continue
#             pj = people_px[j]
#             if pj is None:
#                 continue
#             col = colors[i % len(colors)]
#             cv2.arrowedLine(frame, pi, pj, col, 2, tipLength=0.2)

# def highlight_targets(frame, dets, Gmat, colors):
#     N = len(dets)
#     for i in range(N):
#         for j in range(N):
#             if i == j or Gmat[i, j] == 0:
#                 continue
#             x1, y1, x2, y2 = dets[j]["bbox"]
#             col = colors[i % len(colors)]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

# def project_face_bbox_to_equi(head_box, crop_shape, fov_deg,
#                               r_basis, u_basis, f_basis, W, H, color, frame):
#     ch, cw = crop_shape[:2]
#     corners_crop = [(head_box[0], head_box[1]),
#                     (head_box[2], head_box[1]),
#                     (head_box[2], head_box[3]),
#                     (head_box[0], head_box[3])]
#     corners_eq = [dir_to_equirect_px(
#                     crop_px_to_global_dir(cx, cy, cw, ch, fov_deg,
#                                           r_basis, u_basis, f_basis),
#                     W, H)
#                   for cx, cy in corners_crop]
#     cv2.polylines(frame, [np.array(corners_eq, np.int32)],
#                   isClosed=True, color=color, thickness=2)

# # =========================
# # MAIN
# # =========================
# def main():
#     # CSV
#     session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     csv_path = os.path.join(SAVE_DIR, f"lookat_{session_stamp}.csv")
#     with open(csv_path, "w", newline="") as f:
#         csv.writer(f).writerow(["timestamp", "frame", "person_i", "person_j", "looks_at"])

#     # Video / Camera
#     if USE_CAMERA:
#         cap = cv2.VideoCapture(CAMERA_INDEX)
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
#     else:
#         cap = cv2.VideoCapture(VIDEO_FILE_PATH)
#     if not cap.isOpened():
#         raise RuntimeError("Cannot open input source.")

#     # YOLO
#     yolo = load_yolo(YOLO_MODEL_PATH)

#     last_yolo_t = last_pose_t = 0.0
#     current_dets = []
#     frame_idx = 0

#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break
#         frame_idx += 1
#         H, W = frame.shape[:2]
#         t_now = time.time()
#         draw_frame = frame.copy()

#         # Person detection (interval)
#         if (t_now - last_yolo_t) >= YOLO_INTERVAL_SEC or not current_dets:
#             current_dets = detect_persons(frame, yolo, conf=YOLO_CONF_THRESHOLD, imgsz=YOLO_IMGSZ)
#             current_dets = select_top3_by_size(current_dets)
#             for pid, d in enumerate(current_dets, start=1):
#                 d["id"] = f"P{pid}"
#                 d["color"] = COLORS[(pid-1) % len(COLORS)]
#             last_yolo_t = t_now

#         # Draw person boxes
#         if SHOW_PERSON_BBOX:
#             for d in current_dets:
#                 x1, y1, x2, y2 = d["bbox"]
#                 cv2.rectangle(draw_frame, (x1, y1), (x2, y2), d["color"], 2)
#                 cv2.putText(draw_frame, d["id"], (x1, max(20, y1-10)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, d["color"], 2, cv2.LINE_AA)

#         # Pose + gaze + 3D look-at (interval)
#         if (t_now - last_pose_t) >= POSE_INTERVAL_SEC and current_dets:
#             P_list, G_list, people_ids, dets_used = [], [], [], []

#             for d in current_dets:
#                 u, v = d["center"]

#                 # Gnomonic crop around the person
#                 crop, (r_basis, u_basis, f_basis) = equirect_to_gnomonic(
#                     frame, center_px=(u, v),
#                     fov_deg=GNOMONIC_HFOV_DEG,
#                     out_w=GNOMONIC_OUT_W,
#                     out_h=GNOMONIC_OUT_H,
#                     roll_deg=GNOMONIC_ROLL_DEG
#                 )

#                 pose = estimate_head_pose_with_visuals(crop)
#                 if not pose:
#                     continue

#                 head_box, rvec, lms = pose

#                 # Local forward in crop coords → global gaze vector (for CALCULATIONS)
#                 rmat, _ = cv2.Rodrigues(rvec)
#                 f_vec_crop = rmat @ np.array([0, 0, 1.0], dtype=np.float64)
#                 f_vec_global = _unit(r_basis * f_vec_crop[0] +
#                                      u_basis * f_vec_crop[1] +
#                                      f_basis * f_vec_crop[2])

#                 # Distance from head width (use gnomonic fx)
#                 fx = 0.5 * GNOMONIC_OUT_W / math.tan(math.radians(GNOMONIC_HFOV_DEG/2))
#                 r_dist = estimate_distance_from_headsize(head_box, fx)

#                 # Head 3D position from equirect (ray * distance)
#                 P_i = person_position_from_equi(u, v, W, H, r_dist)

#                 P_list.append(P_i)
#                 G_list.append(f_vec_global)
#                 people_ids.append(d["id"])
#                 dets_used.append(d)

#                 # --- Visualization (Option A): simple, stable 2D arrows in equirect ---
#                 # Treat gnomonic view ~ equirect view for drawing only.
#                 u_vis, v_vis = int(u), int(v)

#                 # Rough yaw/pitch from rotation matrix
#                 rmat_vis, _ = cv2.Rodrigues(rvec)
#                 yaw_vis   = math.atan2(rmat_vis[1, 0], rmat_vis[0, 0])     # left/right
#                 pitch_vis = math.asin(-rmat_vis[2, 0])                     # up/down

#                 arrow_len_px = 120
#                 dx_vis = int(arrow_len_px * math.sin(yaw_vis))
#                 dy_vis = int(-arrow_len_px * math.sin(pitch_vis))
#                 arrow_end_vis = (u_vis + dx_vis, v_vis + dy_vis)

#                 cv2.arrowedLine(draw_frame, (u_vis, v_vis), arrow_end_vis, d["color"], 2, tipLength=0.25)
#                 cv2.putText(draw_frame, d["id"], (u_vis + 10, v_vis - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, d["color"], 2)

#                 # Optional: show projected face bbox poly (helps sanity-check crop projection)
#                 if SHOW_FACE_BBOX:
#                     project_face_bbox_to_equi(head_box, crop.shape, GNOMONIC_HFOV_DEG,
#                                               r_basis, u_basis, f_basis, W, H, d["color"], draw_frame)

#             if len(P_list) >= 2:
#                 Gmat = compute_lookat_matrix(P_list, G_list, deg_thresh=GAZE_THRESH_DEG)
#                 GAZE_HISTORY.append(Gmat.astype(float))
#                 avg_G = np.mean(GAZE_HISTORY, axis=0)  # not shown, but available if needed

#                 print(f"[Frame {frame_idx}] Look-at matrix:\n{Gmat}")

#                 # CSV log
#                 with open(csv_path, "a", newline="") as f:
#                     writer = csv.writer(f)
#                     for i in range(len(P_list)):
#                         for j in range(len(P_list)):
#                             if i != j:
#                                 writer.writerow([datetime.now().isoformat(),
#                                                  frame_idx, i+1, j+1, int(Gmat[i,j])])

#                 # Project 3D head positions back to equirect for arrow anchors
#                 people_px = [project_position_to_equirect_px(P, W, H) for P in P_list]

#                 # Draw IDs at 3D anchors
#                 draw_people_and_ids(draw_frame, people_px, COLORS, people_ids)

#                 # Highlight target bboxes (use only dets we used)
#                 highlight_targets(draw_frame, dets_used, Gmat, COLORS)

#                 # Draw look-at arrows i -> j (between participants)
#                 draw_lookat_arrows(draw_frame, people_px, Gmat, COLORS)

#                 # --- Overlay: look-at relationships & ordering ---
#                 text_y = 40
#                 cv2.putText(draw_frame, "Look-at matrix (Left→Right = P1,P2,P3):",
#                             (50, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#                 for i in range(len(Gmat)):
#                     row_txt = f"P{i+1}: " + " ".join(str(int(x)) for x in Gmat[i])
#                     cv2.putText(draw_frame, row_txt, (50, text_y + (i + 1) * 30),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#                 looking_list = []
#                 for i in range(len(Gmat)):
#                     for j in range(len(Gmat)):
#                         if i != j and Gmat[i, j] == 1:
#                             looking_list.append(f"P{i+1}→P{j+1}")
#                 if looking_list:
#                     summary_text = ", ".join(looking_list)
#                     cv2.putText(draw_frame, summary_text, (50, text_y + 150),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#                 cv2.putText(draw_frame, "Left to Right: P1   P2   P3",
#                             (W//4, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#             last_pose_t = t_now

#         # Show window
#         if SHOW_EQUIRECT_WINDOW:
#             disp_w = min(EQUIRECT_DISPLAY_MAX_W, W)
#             disp_h = int(disp_w / (W / float(H)))
#             equi_disp = cv2.resize(draw_frame, (disp_w, disp_h))
#             cv2.putText(equi_disp, f"Frame: {frame_idx}", (50, disp_h-40),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
#             cv2.imshow("Equirect Preview (3D look-at overlay)", equi_disp)

#             if STEP_MODE:
#                 key = cv2.waitKey(0) & 0xFF  # wait for key each frame
#                 if key == ord('q'):
#                     break
#             else:
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#     cap.release()
#     cv2.destroyAllWindows()
#     print(f"[INFO] Done. CSV → {csv_path}")

# if __name__ == "__main__":
#     main()

######################################

#!/usr/bin/env python3
import os, time, math, csv
from datetime import datetime
from collections import deque, defaultdict
from contextlib import contextmanager

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec

# =========================
# Performance Timing
# =========================
class PerformanceTimer:
    """Track and report performance metrics for each pipeline stage"""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.counts = defaultdict(int)
        self.current_timers = {}
    
    @contextmanager
    def time(self, stage_name):
        """Context manager for timing a code block"""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            self.timings[stage_name].append(elapsed)
            self.counts[stage_name] += 1
    
    def get_stats(self, stage_name):
        """Get statistics for a specific stage"""
        if stage_name not in self.timings or not self.timings[stage_name]:
            return None
        
        times = self.timings[stage_name]
        return {
            'mean': np.mean(times),
            'median': np.median(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'count': self.counts[stage_name],
            'total': np.sum(times)
        }
    
    def get_all_stats(self):
        """Get statistics for all stages"""
        return {stage: self.get_stats(stage) for stage in self.timings.keys()}
    
    def print_summary(self):
        """Print comprehensive performance summary"""
        print("\n" + "="*80)
        print("PERFORMANCE TIMING SUMMARY")
        print("="*80)
        
        all_stats = self.get_all_stats()
        if not all_stats:
            print("No timing data collected.")
            return
        
        # Sort by total time (descending)
        sorted_stages = sorted(all_stats.items(), 
                              key=lambda x: x[1]['total'] if x[1] else 0, 
                              reverse=True)
        
        print(f"\n{'Stage':<30} {'Count':<8} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Total(s)':<10}")
        print("-"*110)
        
        total_time = 0
        for stage, stats in sorted_stages:
            if stats:
                print(f"{stage:<30} {stats['count']:<8} "
                      f"{stats['mean']:>8.2f}ms {stats['median']:>8.2f}ms "
                      f"{stats['std']:>8.2f}ms {stats['min']:>8.2f}ms "
                      f"{stats['max']:>8.2f}ms {stats['total']/1000:>8.2f}s")
                total_time += stats['total']
        
        print("-"*110)
        print(f"{'TOTAL':<30} {'':<8} {'':<10} {'':<10} {'':<10} {'':<10} {'':<10} {total_time/1000:>8.2f}s")
        print("="*80)
        
        # Print bottleneck analysis
        if sorted_stages:
            print("\nBOTTLENECK ANALYSIS:")
            print("-"*80)
            
            # Top 3 by total time
            print("\nTop 3 stages by total time:")
            for i, (stage, stats) in enumerate(sorted_stages[:3], 1):
                if stats:
                    pct = (stats['total'] / total_time) * 100
                    print(f"  {i}. {stage}: {stats['total']/1000:.2f}s ({pct:.1f}%)")
            
            # Stages with high variance
            print("\nStages with high variance (std/mean > 0.5):")
            high_variance = [(stage, stats) for stage, stats in all_stats.items() 
                           if stats and stats['std']/stats['mean'] > 0.5]
            if high_variance:
                for stage, stats in sorted(high_variance, key=lambda x: x[1]['std'], reverse=True)[:3]:
                    cv = stats['std']/stats['mean']
                    print(f"  - {stage}: CV={cv:.2f} (mean={stats['mean']:.2f}ms, std={stats['std']:.2f}ms)")
            else:
                print("  None (all stages have consistent timing)")
            
            print("="*80)
    
    def save_to_csv(self, filepath):
        """Save detailed timing data to CSV"""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Stage', 'Count', 'Mean(ms)', 'Median(ms)', 'Std(ms)', 
                           'Min(ms)', 'Max(ms)', 'Total(s)'])
            
            all_stats = self.get_all_stats()
            for stage in sorted(all_stats.keys()):
                stats = all_stats[stage]
                if stats:
                    writer.writerow([
                        stage,
                        stats['count'],
                        f"{stats['mean']:.2f}",
                        f"{stats['median']:.2f}",
                        f"{stats['std']:.2f}",
                        f"{stats['min']:.2f}",
                        f"{stats['max']:.2f}",
                        f"{stats['total']/1000:.2f}"
                    ])

# =========================
# CONFIG
# =========================
USE_CAMERA = False
# VIDEO_FILE_PATH = r"G:\My Drive\Research\(Shared) Project_mixed_group_interaction\Pilot_data_collection\Session_1\Pilot_1_360_hd_interaction.mp4"
VIDEO_FILE_PATH = r"G:\My Drive\Research\(Shared) Project_mixed_group_interaction\Pilot_data_collection\Session_2\pilot_2_360_hd_interaction_corrected.mp4"
VIDEO_FILE_PATH = r"C:\Users\sures\Documents\Children_school_data\study4\children_school_study_4_activity_interaction.mp4"

# Ground truth labels CSV
GROUND_TRUTH_CSV = r"C:\Users\sures\Documents\Children_school_data\study4\labels_children_school_study_4_combined_p1_p2_p3.csv"

# =========================
# PLAYBACK MODES
# =========================
# Three modes:
# - "debug": Press SPACE to advance frame-by-frame (default)
# - "run": Continuous playback with normal video speed
# - "fast": Continuous playback with minimal delay (for processing)
PLAYBACK_MODE = "fast"  # Options: "debug", "run", "fast"

SAVE_DIR = "live_out"
SHOW_EQUIRECT_WINDOW    = True
EQUIRECT_DISPLAY_MAX_W  = 1280

# Drawing toggles (can be toggled at runtime with keyboard)
SHOW_PERSON_BBOX = True
SHOW_FACE_BBOX   = True
SHOW_FACE_MESH   = True
SHOW_GNOMONIC_CROPS = True
SHOW_GAZE_ARROWS = True      # Show head pose direction arrows
SHOW_GAZE_ANGLES = True      # Show angle measurements between people
SHOW_LOOKAT_ARROWS = True    # Show arrows between people who are looking at each other

# Camera (if using live)
CAMERA_INDEX  = 0
CAM_WIDTH     = 2880
CAM_HEIGHT    = 1440

# YOLO
YOLO_MODEL_PATH     = "yolov8n.pt"
YOLO_CONF_THRESHOLD = 0.35
YOLO_IMGSZ          = 960

# Gnomonic (local synthetic pinhole for person detection)
GNOMONIC_OUT_W    = 512
GNOMONIC_OUT_H    = 512
GNOMONIC_HFOV_DEG = 80.0
GNOMONIC_ROLL_DEG = 180.0

# Face detection & cropping
FACE_CROP_SCALE = 2.5       # Scale factor for face crop (larger = more context)
FACE_CROP_MIN_SIZE = 160    # Minimum face crop size in pixels
FACE_CROP_MAX_SIZE = 400    # Maximum face crop size in pixels

# Social-gaze params
HEAD_WIDTH_M    = 0.135
GAZE_THRESH_DEG = 8.0

# Person colors
COLORS = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,128,0), (255,0,255)]

# Intervals
YOLO_INTERVAL_SEC = 10.0
POSE_INTERVAL_SEC = 0.1

# History smoothing
GAZE_HISTORY = deque(maxlen=30)

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# Ground Truth Loading
# =========================
def load_ground_truth(csv_path, fps=30.0):
    """
    Load ground truth gaze labels from CSV.
    Returns dict mapping frame_num -> set of gaze pairs (e.g., {100: {('P1', 'P3'), ('P2', 'P1')}})
    """
    ground_truth = defaultdict(set)
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row['label_type'].strip()
                
                # Parse PXGazePY labels (e.g., P1GazeP3, P2GazeP1)
                if 'Gaze' in label and label.startswith('P') and not 'Robot' in label and not 'EXPERIMENTER' in label:
                    parts = label.split('Gaze')
                    if len(parts) == 2:
                        looker = parts[0]  # e.g., "P1"
                        target = parts[1]  # e.g., "P3"
                        
                        if looker in ['P1', 'P2', 'P3'] and target in ['P1', 'P2', 'P3']:
                            start_time = float(row['start_time'])
                            end_time = float(row['end_time'])
                            
                            # Convert time to frame numbers
                            start_frame = int(start_time * fps)
                            end_frame = int(end_time * fps)
                            
                            # Mark all frames in this interval
                            for frame_num in range(start_frame, end_frame + 1):
                                ground_truth[frame_num].add((looker, target))
        
        print(f"\n[INFO] Loaded ground truth from {csv_path}")
        print(f"[INFO] Total frames with gaze labels: {len(ground_truth)}")
        
        # Show sample labels
        if ground_truth:
            sample_frames = sorted(ground_truth.keys())[:5]
            print(f"[INFO] Sample labels:")
            for frame in sample_frames:
                print(f"  Frame {frame}: {ground_truth[frame]}")
        
        return ground_truth
    
    except FileNotFoundError:
        print(f"[WARNING] Ground truth file not found: {csv_path}")
        print(f"[WARNING] Continuing without ground truth validation...")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to load ground truth: {e}")
        return None

# =========================
# Validation Metrics
# =========================
class GazeMetrics:
    """Track gaze detection metrics against ground truth"""
    
    def __init__(self):
        # Per-relationship metrics (e.g., 'P1->P2', 'P2->P3', etc.)
        self.metrics = {}
        for i in [1, 2, 3]:
            for j in [1, 2, 3]:
                if i != j:
                    key = f'P{i}->P{j}'
                    self.metrics[key] = {
                        'true_positive': 0,
                        'false_positive': 0,
                        'true_negative': 0,
                        'false_negative': 0
                    }
        
        # Overall metrics
        self.overall = {
            'true_positive': 0,
            'false_positive': 0,
            'true_negative': 0,
            'false_negative': 0
        }
    
    def update(self, frame_num, predicted_gazes, ground_truth_gazes):
        """
        Update metrics for a single frame.
        
        Args:
            frame_num: Frame number
            predicted_gazes: set of tuples, e.g., {('P1', 'P3'), ('P2', 'P1')}
            ground_truth_gazes: set of tuples from ground truth
        """
        # Check all possible gaze relationships
        for i in [1, 2, 3]:
            for j in [1, 2, 3]:
                if i == j:
                    continue
                
                looker = f'P{i}'
                target = f'P{j}'
                key = f'{looker}->{target}'
                
                predicted = (looker, target) in predicted_gazes
                actual = (looker, target) in ground_truth_gazes
                
                # Update confusion matrix
                if predicted and actual:
                    self.metrics[key]['true_positive'] += 1
                    self.overall['true_positive'] += 1
                elif predicted and not actual:
                    self.metrics[key]['false_positive'] += 1
                    self.overall['false_positive'] += 1
                elif not predicted and actual:
                    self.metrics[key]['false_negative'] += 1
                    self.overall['false_negative'] += 1
                else:  # not predicted and not actual
                    self.metrics[key]['true_negative'] += 1
                    self.overall['true_negative'] += 1
    
    def compute_scores(self, metrics_dict):
        """Compute precision, recall, F1, accuracy from confusion matrix"""
        tp = metrics_dict['true_positive']
        fp = metrics_dict['false_positive']
        tn = metrics_dict['true_negative']
        fn = metrics_dict['false_negative']
        
        total = tp + fp + tn + fn
        
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'total': total
        }
    
    def print_summary(self):
        """Print comprehensive metrics summary"""
        print("\n" + "="*80)
        print("GAZE DETECTION VALIDATION SUMMARY")
        print("="*80)
        
        # Overall metrics
        print("\nOVERALL METRICS:")
        overall_scores = self.compute_scores(self.overall)
        print(f"  Accuracy:  {overall_scores['accuracy']:.3f}")
        print(f"  Precision: {overall_scores['precision']:.3f}")
        print(f"  Recall:    {overall_scores['recall']:.3f}")
        print(f"  F1-Score:  {overall_scores['f1']:.3f}")
        print(f"  TP: {overall_scores['tp']}, FP: {overall_scores['fp']}, "
              f"TN: {overall_scores['tn']}, FN: {overall_scores['fn']}")
        
        # Per-relationship metrics
        print("\nPER-RELATIONSHIP METRICS:")
        print("-" * 80)
        print(f"{'Relationship':<12} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} "
              f"{'F1':<10} {'TP':<6} {'FP':<6} {'FN':<6}")
        print("-" * 80)
        
        for key in sorted(self.metrics.keys()):
            scores = self.compute_scores(self.metrics[key])
            if scores['total'] > 0:  # Only show relationships that were evaluated
                print(f"{key:<12} {scores['accuracy']:<10.3f} {scores['precision']:<11.3f} "
                      f"{scores['recall']:<10.3f} {scores['f1']:<10.3f} "
                      f"{scores['tp']:<6} {scores['fp']:<6} {scores['fn']:<6}")
        
        print("="*80 + "\n")
    
    def save_detailed_report(self, filepath):
        """Save detailed metrics to CSV"""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Relationship', 'Accuracy', 'Precision', 'Recall', 'F1', 
                           'TP', 'FP', 'TN', 'FN', 'Total'])
            
            # Overall
            overall_scores = self.compute_scores(self.overall)
            writer.writerow(['OVERALL', 
                           f"{overall_scores['accuracy']:.4f}",
                           f"{overall_scores['precision']:.4f}",
                           f"{overall_scores['recall']:.4f}",
                           f"{overall_scores['f1']:.4f}",
                           overall_scores['tp'],
                           overall_scores['fp'],
                           overall_scores['tn'],
                           overall_scores['fn'],
                           overall_scores['total']])
            
            writer.writerow([])  # Empty row
            
            # Per-relationship
            for key in sorted(self.metrics.keys()):
                scores = self.compute_scores(self.metrics[key])
                if scores['total'] > 0:
                    writer.writerow([key,
                                   f"{scores['accuracy']:.4f}",
                                   f"{scores['precision']:.4f}",
                                   f"{scores['recall']:.4f}",
                                   f"{scores['f1']:.4f}",
                                   scores['tp'],
                                   scores['fp'],
                                   scores['tn'],
                                   scores['fn'],
                                   scores['total']])

# =========================
# MediaPipe Face Detection (Stage 1: Fast face detector)
# =========================
# Model 0: short-range (up to 2m), Model 1: full-range (up to 5m)
mp_face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=1,  # Full-range model
    min_detection_confidence=0.5  # This is explicitly documented!
)

# =========================
# MediaPipe FaceMesh (Stage 2: Detailed landmarks & pose)
# =========================
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True,
    min_detection_confidence=0.5,  # Documented threshold
    min_tracking_confidence=0.5    # Documented threshold
)

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Custom drawing specs
FACE_MESH_TESSELATION = DrawingSpec(color=(128, 128, 128), thickness=1, circle_radius=1)
FACE_MESH_CONTOURS = DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
FACE_MESH_IRISES = DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)

# PnP model points
FACE_3D_IDXS = [1, 152, 33, 263, 61, 291]
FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1)
], dtype=np.float64)

# =========================
# Geometry helpers
# =========================
def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / (n + 1e-9)

def px_to_dir_equirect(u, v, W, H):
    lon = (u / float(W)) * 2.0 * math.pi - math.pi
    lat = math.pi/2.0 - (v / float(H)) * math.pi
    x = math.cos(lat) * math.cos(lon)
    y = math.sin(lat)
    z = math.cos(lat) * math.sin(lon)
    return np.array([x, y, z], dtype=np.float64)

def dir_to_equirect_px(D, W, H):
    Dx, Dy, Dz = D
    lam = math.atan2(Dz, Dx)
    phi = math.asin(np.clip(Dy, -1.0, 1.0))
    u = (lam + math.pi) / (2 * math.pi) * W
    v = (math.pi/2 - phi) / math.pi * H
    return int(round(u)), int(round(v))

def _dir_from_lonlat(lam, phi):
    c = np.cos(phi)
    return np.array([c*np.cos(lam), np.sin(phi), c*np.sin(lam)], dtype=np.float64)

def _lonlat_from_px(u, v, W, H):
    lam = 2*math.pi*(u/W) - math.pi
    phi = math.pi/2 - math.pi*(v/H)
    return lam, phi

def _basis_from_forward(f):
    f = f/np.linalg.norm(f)
    up = np.array([0, 1, 0.0], dtype=np.float64)
    u = up - f*np.dot(up, f)
    if np.linalg.norm(u) < 1e-9:
        up = np.array([1, 0, 0.0], dtype=np.float64)
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

def equirect_to_gnomonic(equi_bgr, *, center_px, fov_deg, out_w, out_h, roll_deg):
    H, W = equi_bgr.shape[:2]
    lam, phi = _lonlat_from_px(center_px[0], center_px[1], W, H)
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

    crop = cv2.remap(equi_bgr, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return crop, (r, u, f)

def crop_px_to_global_dir(px, py, crop_w, crop_h, fov_deg, r_vec, u_vec, f_vec):
    x_ndc = (px / crop_w - 0.5) * 2
    y_ndc = (py / crop_h - 0.5) * 2
    hfov = math.radians(fov_deg)
    vfov = 2 * math.atan(math.tan(hfov/2) * (crop_h/crop_w))
    lx = math.tan(hfov/2) * x_ndc
    ly = math.tan(vfov/2) * y_ndc
    lz = 1.0
    dir_cam = np.array([lx, ly, lz], dtype=np.float64)
    dir_cam /= np.linalg.norm(dir_cam)
    return _unit(r_vec*dir_cam[0] + u_vec*dir_cam[1] + f_vec*dir_cam[2])

# =========================
# TWO-STAGE FACE DETECTION
# =========================

def detect_face_simple(image):
    """
    Stage 1: Fast face detection using MediaPipe Face Detection
    Returns: (face_bbox, confidence) or None
    """
    h, w = image.shape[:2]
    
    # MediaPipe expects RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_face_detection.process(rgb)
    
    if not results.detections:
        return None
    
    # Get the first (most confident) detection
    detection = results.detections[0]
    
    # Extract bounding box
    bbox = detection.location_data.relative_bounding_box
    x1 = int(bbox.xmin * w)
    y1 = int(bbox.ymin * h)
    x2 = int((bbox.xmin + bbox.width) * w)
    y2 = int((bbox.ymin + bbox.height) * h)
    
    # Get confidence score
    confidence = detection.score[0]
    
    return {
        'bbox': [x1, y1, x2, y2],
        'confidence': confidence,
        'center': [(x1 + x2) / 2, (y1 + y2) / 2]
    }

def crop_face_with_margin(image, face_bbox, scale=FACE_CROP_SCALE, 
                          min_size=FACE_CROP_MIN_SIZE, max_size=FACE_CROP_MAX_SIZE):
    """
    Crop image around detected face with margin for context.
    
    Args:
        image: Input image
        face_bbox: [x1, y1, x2, y2] face bounding box
        scale: How much larger than face bbox (2.5 = 2.5x face size)
        min_size: Minimum crop dimension
        max_size: Maximum crop dimension
    
    Returns:
        cropped_image, crop_info dict
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = face_bbox
    
    # Face dimensions
    face_w = x2 - x1
    face_h = y2 - y1
    face_cx = (x1 + x2) / 2
    face_cy = (y1 + y2) / 2
    
    # Calculate crop size (make it square, use larger dimension)
    crop_size = max(face_w, face_h) * scale
    crop_size = max(min_size, min(max_size, crop_size))  # Clamp to min/max
    crop_size = int(crop_size)
    
    # Calculate crop bounds (centered on face)
    half = crop_size // 2
    cx1 = max(0, int(face_cx - half))
    cy1 = max(0, int(face_cy - half))
    cx2 = min(w, int(face_cx + half))
    cy2 = min(h, int(face_cy + half))
    
    # Adjust if crop goes out of bounds
    if cx2 - cx1 < crop_size:
        if cx1 == 0:
            cx2 = min(w, cx1 + crop_size)
        else:
            cx1 = max(0, cx2 - crop_size)
    
    if cy2 - cy1 < crop_size:
        if cy1 == 0:
            cy2 = min(h, cy1 + crop_size)
        else:
            cy1 = max(0, cy2 - crop_size)
    
    # Perform crop
    cropped = image[cy1:cy2, cx1:cx2].copy()
    
    crop_info = {
        'bbox': [cx1, cy1, cx2, cy2],
        'original_face_bbox': face_bbox,
        'crop_size': crop_size,
        'offset': (cx1, cy1)  # Offset for mapping coordinates back
    }
    
    return cropped, crop_info

def estimate_head_pose_two_stage(person_crop, perf_timer=None):
    """
    Two-stage approach:
    1. Detect face with simple detector
    2. Crop tightly around face
    3. Run FaceMesh on face crop for detailed landmarks
    
    Returns: (head_box, rvec, lms, debug_info, annotated_crop, face_crop_vis, crop_info, face_center_in_person) or None
    """
    debug_info = {
        "stage1_face_detected": False,
        "stage1_confidence": None,
        "stage2_facemesh_detected": False,
        "face_crop_size": None,
        "original_crop_size": person_crop.shape[:2]
    }
    
    # STAGE 1: Fast face detection
    if perf_timer:
        with perf_timer.time("4a. Stage1: Face Detection"):
            face_det = detect_face_simple(person_crop)
    else:
        face_det = detect_face_simple(person_crop)
    
    if face_det is None:
        return None
    
    debug_info["stage1_face_detected"] = True
    debug_info["stage1_confidence"] = face_det['confidence']
    
    # Calculate face center in person crop coordinates
    face_center_in_person = face_det['center']
    
    # STAGE 2: Crop around face with margin
    if perf_timer:
        with perf_timer.time("4b. Stage2: Face Cropping"):
            face_crop, crop_info = crop_face_with_margin(person_crop, face_det['bbox'])
    else:
        face_crop, crop_info = crop_face_with_margin(person_crop, face_det['bbox'])
    
    debug_info["face_crop_size"] = face_crop.shape[:2]
    
    # Visualize the face crop
    face_crop_vis = face_crop.copy()
    
    # STAGE 3: Run FaceMesh on the tight face crop
    h, w = face_crop.shape[:2]
    
    if h < 100 or w < 100:
        return None
    
    if perf_timer:
        with perf_timer.time("4c. Stage3: FaceMesh Processing"):
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            results = mp_face_mesh.process(rgb)
    else:
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb)
    
    if not results.multi_face_landmarks:
        return None
    
    debug_info["stage2_facemesh_detected"] = True
    
    lms = results.multi_face_landmarks[0]
    
    # Draw face mesh on face crop
    if SHOW_FACE_MESH:
        mp_drawing.draw_landmarks(
            image=face_crop_vis,
            landmark_list=lms,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=FACE_MESH_TESSELATION,
            connection_drawing_spec=FACE_MESH_TESSELATION
        )
        mp_drawing.draw_landmarks(
            image=face_crop_vis,
            landmark_list=lms,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=FACE_MESH_CONTOURS
        )
        mp_drawing.draw_landmarks(
            image=face_crop_vis,
            landmark_list=lms,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=FACE_MESH_IRISES
        )
    
    # Get bounding box in face crop coordinates
    xs, ys = [], []
    for lm in lms.landmark:
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))
    head_box_in_crop = (min(xs), min(ys), max(xs), max(ys))
    
    # Draw bbox on face crop
    cv2.rectangle(face_crop_vis, (head_box_in_crop[0], head_box_in_crop[1]), 
                 (head_box_in_crop[2], head_box_in_crop[3]), (0, 255, 0), 2)
    
    # PnP solve
    if perf_timer:
        with perf_timer.time("4d. Stage4: PnP Head Pose"):
            face_2d = np.array([(lms.landmark[i].x * w, lms.landmark[i].y * h) 
                                for i in FACE_3D_IDXS], dtype=np.float64)
            cam_matrix = get_camera_matrix(w, h)
            dist = np.zeros((4, 1), dtype=np.float64)
            success, rvec, tvec = cv2.solvePnP(FACE_3D_MODEL, face_2d, cam_matrix, 
                                                dist, flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        face_2d = np.array([(lms.landmark[i].x * w, lms.landmark[i].y * h) 
                            for i in FACE_3D_IDXS], dtype=np.float64)
        cam_matrix = get_camera_matrix(w, h)
        dist = np.zeros((4, 1), dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(FACE_3D_MODEL, face_2d, cam_matrix, 
                                            dist, flags=cv2.SOLVEPNP_ITERATIVE)
    
    if not success:
        return None
    
    # Draw pose axes on face crop
    if SHOW_FACE_MESH:
        nose_tip = (int(face_2d[0][0]), int(face_2d[0][1]))
        axis_length = 50
        axis_3d = np.float32([[axis_length, 0, 0],
                              [0, axis_length, 0],
                              [0, 0, axis_length]])
        axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, cam_matrix, dist)
        axis_2d = axis_2d.reshape(-1, 2).astype(int)
        
        cv2.line(face_crop_vis, nose_tip, tuple(axis_2d[0]), (0, 0, 255), 3)
        cv2.line(face_crop_vis, nose_tip, tuple(axis_2d[1]), (0, 255, 0), 3)
        cv2.line(face_crop_vis, nose_tip, tuple(axis_2d[2]), (255, 0, 0), 3)
    
    # Map head_box back to person_crop coordinates
    cx1, cy1 = crop_info['offset']
    head_box_in_person = (
        head_box_in_crop[0] + cx1,
        head_box_in_crop[1] + cy1,
        head_box_in_crop[2] + cx1,
        head_box_in_crop[3] + cy1
    )
    
    # Also draw Stage 1 face detection box on person crop for comparison
    annotated_person = person_crop.copy()
    cv2.rectangle(annotated_person, 
                 (face_det['bbox'][0], face_det['bbox'][1]),
                 (face_det['bbox'][2], face_det['bbox'][3]),
                 (255, 0, 255), 2)  # Magenta for Stage 1
    cv2.putText(annotated_person, "Stage1: Face Det", 
               (face_det['bbox'][0], face_det['bbox'][1]-5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    return (head_box_in_person, rvec, lms, debug_info, annotated_person, 
            face_crop_vis, crop_info, face_center_in_person)

# =========================
# Head pose estimation helpers
# =========================
def get_camera_matrix(w, h):
    focal_length = w
    center = (w / 2, h / 2)
    return np.array([[focal_length, 0, center[0]],
                     [0, focal_length, center[1]],
                     [0, 0, 1]], dtype="double")

# =========================
# YOLO person detection
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
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        out.append({"bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "center": [u, v], "conf": float(c)})
    return out

def select_top3_by_size(detections):
    dets_sorted = sorted(detections,
                         key=lambda d: (d["bbox"][2]-d["bbox"][0]) * (d["bbox"][3]-d["bbox"][1]),
                         reverse=True)
    return dets_sorted[:3]

def assign_ids_left_to_right(detections):
    """
    Assign P1, P2, P3 IDs based on left-to-right position in equirectangular image.
    Uses center x-coordinate of each person's bounding box.
    """
    if not detections:
        return detections
    
    # Sort by x-center (left to right)
    sorted_dets = sorted(detections, key=lambda d: d["center"][0])
    
    # Assign IDs P1, P2, P3 from left to right
    for pid, d in enumerate(sorted_dets, start=1):
        d["id"] = f"P{pid}"
        d["color"] = COLORS[(pid-1) % len(COLORS)]
    
    return sorted_dets

# =========================
# 3D look-at geometry
# =========================
def estimate_distance_from_headsize(head_bbox, fx, head_width_m=HEAD_WIDTH_M):
    w_px = max(head_bbox[2] - head_bbox[0], 1e-6)
    return (fx * head_width_m) / w_px

def person_position_from_equi(u, v, W, H, r):
    D = px_to_dir_equirect(u, v, W, H)
    return r * (D / np.linalg.norm(D))

def looks_at(P_i, G_i, P_j, deg_thresh=GAZE_THRESH_DEG):
    T = P_j - P_i
    nT = np.linalg.norm(T)
    if nT < 1e-6:
        return False
    cosang = np.dot(G_i, T / nT)
    return cosang >= math.cos(math.radians(deg_thresh))

def compute_gaze_angle(P_i, G_i, P_j):
    """
    Compute angle in degrees between person i's gaze direction and the direction to person j.
    Returns None if computation fails.
    """
    T = P_j - P_i
    nT = np.linalg.norm(T)
    if nT < 1e-6:
        return None
    cosang = np.dot(G_i, T / nT)
    cosang = np.clip(cosang, -1.0, 1.0)
    angle_rad = math.acos(cosang)
    return math.degrees(angle_rad)

def compute_lookat_matrix(P_list, G_list, deg_thresh=GAZE_THRESH_DEG):
    N = len(P_list)
    Gmat = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            if i != j and looks_at(P_list[i], G_list[i], P_list[j], deg_thresh):
                Gmat[i, j] = 1
    return Gmat

def compute_angle_matrix(P_list, G_list):
    """
    Compute matrix of angles between each person's gaze and every other person.
    Returns NxN matrix where entry [i,j] is angle from person i's gaze to person j.
    """
    N = len(P_list)
    angle_mat = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i != j:
                angle = compute_gaze_angle(P_list[i], G_list[i], P_list[j])
                if angle is not None:
                    angle_mat[i, j] = angle
                else:
                    angle_mat[i, j] = -1.0  # Invalid
    return angle_mat

# =========================
# Visualization helpers
# =========================
def project_position_to_equirect_px(P, W, H):
    if np.linalg.norm(P) < 1e-9:
        return None
    D = _unit(P)
    return dir_to_equirect_px(D, W, H)

def draw_people_and_ids(frame, people_px, colors, ids):
    for idx, p in enumerate(people_px):
        if p is None:
            continue
        u, v = p
        col = colors[idx % len(colors)]
        cv2.circle(frame, (u, v), 8, col, -1)
        cv2.putText(frame, ids[idx], (u+10, max(20, v-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

def draw_lookat_arrows(frame, people_px, Gmat, colors, show_arrows=True):
    """Draw arrows from looker to target when gaze detected"""
    if not show_arrows:
        return
    
    N = len(people_px)
    for i in range(N):
        pi = people_px[i]
        if pi is None:
            continue
        for j in range(N):
            if i == j or Gmat[i, j] == 0:
                continue
            pj = people_px[j]
            if pj is None:
                continue
            col = colors[i % len(colors)]
            cv2.arrowedLine(frame, pi, pj, col, 2, tipLength=0.2)

def draw_gaze_angles(frame, people_px, angle_mat, colors, ids):
    """Draw angle measurements between people's gazes"""
    N = len(people_px)
    for i in range(N):
        pi = people_px[i]
        if pi is None:
            continue
        for j in range(N):
            if i == j:
                continue
            pj = people_px[j]
            if pj is None or angle_mat[i, j] < 0:
                continue
            
            # Draw angle text at midpoint between people
            mid_x = int((pi[0] + pj[0]) / 2)
            mid_y = int((pi[1] + pj[1]) / 2)
            
            angle_deg = angle_mat[i, j]
            angle_text = f"{ids[i]}->{ids[j]}: {angle_deg:.1f}°"
            
            # Color code: green if looking (< threshold), red otherwise
            text_color = (0, 255, 0) if angle_deg < GAZE_THRESH_DEG else (0, 0, 255)
            
            # Draw background rectangle for text
            text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, 
                         (mid_x - 5, mid_y - text_size[1] - 5),
                         (mid_x + text_size[0] + 5, mid_y + 5),
                         (0, 0, 0), -1)
            
            cv2.putText(frame, angle_text, (mid_x, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

def highlight_targets(frame, dets, Gmat, colors):
    N = len(dets)
    for i in range(N):
        for j in range(N):
            if i == j or Gmat[i, j] == 0:
                continue
            x1, y1, x2, y2 = dets[j]["bbox"]
            col = colors[i % len(colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

def project_face_bbox_to_equi(head_box, crop_shape, fov_deg,
                              r_basis, u_basis, f_basis, W, H, color, frame):
    ch, cw = crop_shape[:2]
    corners_crop = [(head_box[0], head_box[1]),
                    (head_box[2], head_box[1]),
                    (head_box[2], head_box[3]),
                    (head_box[0], head_box[3])]
    corners_eq = [dir_to_equirect_px(
                    crop_px_to_global_dir(cx, cy, cw, ch, fov_deg,
                                          r_basis, u_basis, f_basis),
                    W, H)
                  for cx, cy in corners_crop]
    cv2.polylines(frame, [np.array(corners_eq, np.int32)],
                  isClosed=True, color=color, thickness=2)

# =========================
# MAIN
# =========================
def main():
    # Initialize performance timer
    perf_timer = PerformanceTimer()
    
    # Runtime toggles (can be changed with keyboard)
    show_person_bbox = SHOW_PERSON_BBOX
    show_face_bbox = SHOW_FACE_BBOX
    show_face_mesh = SHOW_FACE_MESH
    show_gnomonic_crops = SHOW_GNOMONIC_CROPS
    show_gaze_arrows = SHOW_GAZE_ARROWS
    show_gaze_angles = SHOW_GAZE_ANGLES
    show_lookat_arrows = SHOW_LOOKAT_ARROWS
    
    playback_mode = PLAYBACK_MODE
    
    # Load ground truth
    cap_temp = cv2.VideoCapture(VIDEO_FILE_PATH)
    fps = cap_temp.get(cv2.CAP_PROP_FPS) or 30.0
    cap_temp.release()
    
    ground_truth = load_ground_truth(GROUND_TRUTH_CSV, fps=fps)
    metrics = GazeMetrics() if ground_truth else None
    
    # CSV
    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(SAVE_DIR, f"lookat_{session_stamp}.csv")
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["timestamp", "frame", "person_i", "person_j", "looks_at", 
                  "stage1_conf_i", "stage1_conf_j", "face_detected_i", "face_detected_j"]
        if ground_truth:
            header.extend(["ground_truth", "correct_prediction"])
        writer.writerow(header)

    # Video / Camera
    if USE_CAMERA:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    else:
        cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input source.")

    # YOLO
    yolo = load_yolo(YOLO_MODEL_PATH)

    last_yolo_t = last_pose_t = 0.0
    current_dets = []
    frame_idx = 0
    
    # Statistics
    total_frames = 0
    stage1_attempts = 0
    stage1_success = 0
    stage2_attempts = 0
    stage2_success = 0

    print("\n" + "="*80)
    print("TWO-STAGE FACE DETECTION WITH GROUND TRUTH VALIDATION")
    print("="*80)
    print(f"Playback Mode: {playback_mode.upper()}")
    print(f"  - 'debug': Press SPACE to advance frame-by-frame")
    print(f"  - 'run': Continuous playback at video speed")
    print(f"  - 'fast': Continuous playback with minimal delay")
    print(f"\nVideo FPS: {fps:.2f}")
    print(f"Gaze threshold: {GAZE_THRESH_DEG}°")
    if ground_truth:
        print(f"Ground truth: LOADED ({len(ground_truth)} labeled frames)")
    else:
        print(f"Ground truth: NOT LOADED")
    print("\n" + "-"*80)
    print("KEYBOARD CONTROLS:")
    print("  Q         - Quit")
    print("  SPACE     - Next frame (in debug mode) / Pause-Resume (in run/fast modes)")
    print("  S         - Save current frame")
    print("  M         - Cycle playback mode (debug → run → fast)")
    print("  1         - Toggle person bounding boxes")
    print("  2         - Toggle face bounding boxes")
    print("  3         - Toggle face mesh visualization")
    print("  4         - Toggle gnomonic crop windows")
    print("  5         - Toggle gaze direction arrows")
    print("  6         - Toggle gaze angle measurements")
    print("  7         - Toggle look-at connection arrows")
    print("="*80 + "\n")
    
    paused = False

    while True:
        # Handle pause in run/fast modes
        if paused and playback_mode in ["run", "fast"]:
            # Display current frame while paused
            if SHOW_EQUIRECT_WINDOW:
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = False
                    print(f"[INFO] Resumed")
                elif key == ord('m') or key == ord('M'):
                    modes = ["debug", "run", "fast"]
                    current_idx = modes.index(playback_mode)
                    playback_mode = modes[(current_idx + 1) % len(modes)]
                    print(f"[INFO] Switched to {playback_mode.upper()} mode")
                    if playback_mode == "debug":
                        paused = False
            continue
        
        with perf_timer.time("1. Frame Capture"):
            ok, frame = cap.read()
            if not ok:
                break
        frame_idx += 1
        total_frames += 1
        H, W = frame.shape[:2]
        t_now = time.time()
        draw_frame = frame.copy()

        # Person detection (interval)
        if (t_now - last_yolo_t) >= YOLO_INTERVAL_SEC or not current_dets:
            with perf_timer.time("2. YOLO Person Detection"):
                current_dets = detect_persons(frame, yolo, conf=YOLO_CONF_THRESHOLD, imgsz=YOLO_IMGSZ)
                current_dets = select_top3_by_size(current_dets)
                current_dets = assign_ids_left_to_right(current_dets)
            last_yolo_t = t_now

        # Draw person boxes
        if show_person_bbox:
            for d in current_dets:
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), d["color"], 2)
                cv2.putText(draw_frame, d["id"], (x1, max(20, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, d["color"], 2, cv2.LINE_AA)

        # Pose + gaze (interval)
        if (t_now - last_pose_t) >= POSE_INTERVAL_SEC and current_dets:
            P_list, G_list, people_ids, dets_used = [], [], [], []
            debug_infos = []
            crop_visuals = []

            for d in current_dets:
                u, v = d["center"]
                stage1_attempts += 1

                # Gnomonic crop around person
                with perf_timer.time("3. Gnomonic Projection"):
                    crop, (r_basis, u_basis, f_basis) = equirect_to_gnomonic(
                        frame, center_px=(u, v),
                        fov_deg=GNOMONIC_HFOV_DEG,
                        out_w=GNOMONIC_OUT_W,
                        out_h=GNOMONIC_OUT_H,
                        roll_deg=GNOMONIC_ROLL_DEG
                    )

                # TWO-STAGE face detection & pose estimation
                with perf_timer.time("4. Face Detection (Total)"):
                    pose_result = estimate_head_pose_two_stage(crop, perf_timer)
                
                if pose_result is None:
                    continue
                
                head_box, rvec, lms, debug_info, annotated_crop, face_crop_vis, crop_info, face_center_in_crop = pose_result
                
                if debug_info["stage1_face_detected"]:
                    stage1_success += 1
                if debug_info["stage2_facemesh_detected"]:
                    stage2_attempts += 1
                    stage2_success += 1

                # Store visualizations
                if show_gnomonic_crops:
                    combined_vis = np.hstack([
                        cv2.resize(annotated_crop, (256, 256)),
                        cv2.resize(face_crop_vis, (256, 256))
                    ])
                    cv2.putText(combined_vis, f"{d['id']} - Person | Face", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    crop_visuals.append((d['id'], combined_vis))

                # Global gaze vector for 3D calculations
                rmat, _ = cv2.Rodrigues(rvec)
                f_vec_crop = rmat @ np.array([0, 0, 1.0], dtype=np.float64)
                f_vec_global = _unit(r_basis * f_vec_crop[0] +
                                     u_basis * f_vec_crop[1] +
                                     f_basis * f_vec_crop[2])

                # Distance estimation
                fx = 0.5 * GNOMONIC_OUT_W / math.tan(math.radians(GNOMONIC_HFOV_DEG/2))
                r_dist = estimate_distance_from_headsize(head_box, fx)

                # Map face center
                face_dir_global = crop_px_to_global_dir(
                    face_center_in_crop[0], face_center_in_crop[1],
                    GNOMONIC_OUT_W, GNOMONIC_OUT_H, GNOMONIC_HFOV_DEG,
                    r_basis, u_basis, f_basis
                )
                face_u, face_v = dir_to_equirect_px(face_dir_global, W, H)
                d["face_position"] = (face_u, face_v)

                # 3D position
                P_i = person_position_from_equi(face_u, face_v, W, H, r_dist)

                P_list.append(P_i)
                G_list.append(f_vec_global)
                people_ids.append(d["id"])
                dets_used.append(d)
                debug_infos.append(debug_info)

                # Draw head pose arrow (gaze direction)
                if show_gaze_arrows:
                    with perf_timer.time("5. Gaze Arrow Rendering"):
                        rmat_vis, _ = cv2.Rodrigues(rvec)
                        pitch_vis = math.asin(-rmat_vis[2, 0])
                        yaw_vis = math.atan2(rmat_vis[2, 1], rmat_vis[2, 2])
                        yaw_vis = -yaw_vis
                        pitch_vis = -pitch_vis
                        
                        arrow_len_px = 120
                        dx_vis = int(arrow_len_px * math.sin(yaw_vis))
                        dy_vis = int(-arrow_len_px * math.sin(pitch_vis))
                        arrow_end_vis = (face_u + dx_vis, face_v + dy_vis)

                        cv2.arrowedLine(draw_frame, (face_u, face_v), arrow_end_vis, 
                                       d["color"], 3, tipLength=0.25)
                        cv2.circle(draw_frame, (face_u, face_v), 6, d["color"], -1)
                        cv2.circle(draw_frame, (face_u, face_v), 8, (255, 255, 255), 2)

                if show_face_bbox:
                    project_face_bbox_to_equi(head_box, crop.shape, GNOMONIC_HFOV_DEG,
                                              r_basis, u_basis, f_basis, W, H, d["color"], draw_frame)

            # Display crop visuals
            if show_gnomonic_crops and crop_visuals:
                for idx, (pid, crop_img) in enumerate(crop_visuals):
                    cv2.imshow(f"Person {pid} - 2-Stage", crop_img)
            elif not show_gnomonic_crops:
                # Close crop windows if toggle is off
                for pid in ['P1', 'P2', 'P3']:
                    try:
                        cv2.destroyWindow(f"Person {pid} - 2-Stage")
                    except:
                        pass

            # Look-at analysis
            if len(P_list) >= 2:
                with perf_timer.time("6. Look-at Matrix Computation"):
                    Gmat = compute_lookat_matrix(P_list, G_list, deg_thresh=GAZE_THRESH_DEG)
                    angle_mat = compute_angle_matrix(P_list, G_list)
                
                # Convert Gmat to predicted gaze set
                predicted_gazes = set()
                for i in range(len(P_list)):
                    for j in range(len(P_list)):
                        if i != j and Gmat[i, j] == 1:
                            predicted_gazes.add((people_ids[i], people_ids[j]))
                
                # Get ground truth for this frame
                gt_gazes = ground_truth.get(frame_idx, set()) if ground_truth else set()
                
                # Update metrics
                if metrics and gt_gazes:  # Only update if we have ground truth for this frame
                    with perf_timer.time("7. Ground Truth Validation"):
                        metrics.update(frame_idx, predicted_gazes, gt_gazes)

                # CSV logging
                with perf_timer.time("8. CSV Logging"):
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        for i in range(len(P_list)):
                            for j in range(len(P_list)):
                                if i != j:
                                    predicted = int(Gmat[i,j])
                                    row = [
                                        datetime.now().isoformat(),
                                        frame_idx, i+1, j+1, predicted,
                                        debug_infos[i]["stage1_confidence"],
                                        debug_infos[j]["stage1_confidence"],
                                        debug_infos[i]["stage2_facemesh_detected"],
                                        debug_infos[j]["stage2_facemesh_detected"]
                                    ]
                                    
                                    if ground_truth:
                                        gaze_pair = (people_ids[i], people_ids[j])
                                        actual = 1 if gaze_pair in gt_gazes else 0
                                        correct = 1 if predicted == actual else 0
                                        row.extend([actual, correct])
                                    
                                    writer.writerow(row)

                with perf_timer.time("9. Visualization Rendering"):
                    people_px = [project_position_to_equirect_px(P, W, H) for P in P_list]
                    draw_people_and_ids(draw_frame, people_px, COLORS, people_ids)
                    highlight_targets(draw_frame, dets_used, Gmat, COLORS)
                    
                    # Draw look-at arrows (person to person connections)
                    if show_lookat_arrows:
                        draw_lookat_arrows(draw_frame, people_px, Gmat, COLORS, show_arrows=True)
                    
                    # Draw gaze angle measurements
                    if show_gaze_angles:
                        draw_gaze_angles(draw_frame, people_px, angle_mat, COLORS, people_ids)

                # Overlay text
                text_y = 40
                cv2.putText(draw_frame, "Predicted Look-at:",
                            (50, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                for i in range(len(Gmat)):
                    row_txt = f"P{i+1}: " + " ".join(str(int(x)) for x in Gmat[i])
                    cv2.putText(draw_frame, row_txt, (50, text_y + (i + 1) * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Show ground truth if available
                if ground_truth and frame_idx in ground_truth:
                    gt_text_y = text_y + 120
                    cv2.putText(draw_frame, f"Ground Truth (Frame {frame_idx}):",
                               (50, gt_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    gt_list = sorted(list(gt_gazes))
                    for idx, (looker, target) in enumerate(gt_list):
                        cv2.putText(draw_frame, f"{looker}→{target}",
                                   (50, gt_text_y + (idx + 1) * 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            last_pose_t = t_now

        # Display left-to-right ordering
        if current_dets:
            left_to_right = " | ".join([d['id'] for d in current_dets])
            cv2.putText(draw_frame, f"Left → Right: {left_to_right}",
                        (W//4, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display statistics
        if stage1_attempts > 0:
            s1_rate = (stage1_success / stage1_attempts) * 100
            stats_text = f"Stage1(FaceDet): {stage1_success}/{stage1_attempts} ({s1_rate:.1f}%)"
            cv2.putText(draw_frame, stats_text, (50, H - 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if stage2_attempts > 0:
            s2_rate = (stage2_success / stage2_attempts) * 100
            stats_text2 = f"Stage2(FaceMesh): {stage2_success}/{stage2_attempts} ({s2_rate:.1f}%)"
            cv2.putText(draw_frame, stats_text2, (50, H - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display mode and controls
        mode_text = f"Mode: {playback_mode.upper()}"
        if paused:
            mode_text += " [PAUSED]"
        cv2.putText(draw_frame, mode_text, (W - 300, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display toggle states
        toggles_y = H - 150
        toggle_text = f"Toggles: "
        if show_person_bbox: toggle_text += "1:Box "
        if show_face_bbox: toggle_text += "2:Face "
        if show_face_mesh: toggle_text += "3:Mesh "
        if show_gnomonic_crops: toggle_text += "4:Crops "
        if show_gaze_arrows: toggle_text += "5:Gaze "
        if show_gaze_angles: toggle_text += "6:Angles "
        if show_lookat_arrows: toggle_text += "7:LookAt"
        cv2.putText(draw_frame, toggle_text, (50, toggles_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Display
        if SHOW_EQUIRECT_WINDOW:
            with perf_timer.time("10. Display Rendering"):
                disp_w = min(EQUIRECT_DISPLAY_MAX_W, W)
                disp_h = int(disp_w / (W / float(H)))
                equi_disp = cv2.resize(draw_frame, (disp_w, disp_h))
                cv2.putText(equi_disp, f"Frame: {frame_idx}", (50, disp_h-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.imshow("Equirect - Two-Stage Detection", equi_disp)

            # Keyboard handling based on playback mode
            if playback_mode == "debug":
                # Debug mode: wait for spacebar or other key
                key = cv2.waitKey(0) & 0xFF
            elif playback_mode == "run":
                # Run mode: normal video speed
                frame_delay = int(1000 / fps) if fps > 0 else 33
                key = cv2.waitKey(frame_delay) & 0xFF
            else:  # fast mode
                # Fast mode: minimal delay
                key = cv2.waitKey(1) & 0xFF
            
            # Handle keyboard input
            if key == ord('q'):
                print("\n[INFO] Quitting...")
                break
            elif key == ord('s'):
                save_path = os.path.join(SAVE_DIR, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(save_path, draw_frame)
                print(f"[INFO] Saved frame to {save_path}")
            elif key == ord('m') or key == ord('M'):
                # Cycle through modes
                modes = ["debug", "run", "fast"]
                current_idx = modes.index(playback_mode)
                playback_mode = modes[(current_idx + 1) % len(modes)]
                print(f"[INFO] Switched to {playback_mode.upper()} mode")
            elif key == ord(' '):
                # Space: in debug mode advances frame (automatic), in run/fast modes toggles pause
                if playback_mode in ["run", "fast"]:
                    paused = not paused
                    print(f"[INFO] {'Paused' if paused else 'Resumed'}")
            elif key == ord('1'):
                show_person_bbox = not show_person_bbox
                print(f"[INFO] Person bbox: {'ON' if show_person_bbox else 'OFF'}")
            elif key == ord('2'):
                show_face_bbox = not show_face_bbox
                print(f"[INFO] Face bbox: {'ON' if show_face_bbox else 'OFF'}")
            elif key == ord('3'):
                show_face_mesh = not show_face_mesh
                print(f"[INFO] Face mesh: {'ON' if show_face_mesh else 'OFF'}")
            elif key == ord('4'):
                show_gnomonic_crops = not show_gnomonic_crops
                print(f"[INFO] Gnomonic crops: {'ON' if show_gnomonic_crops else 'OFF'}")
            elif key == ord('5'):
                show_gaze_arrows = not show_gaze_arrows
                print(f"[INFO] Gaze arrows: {'ON' if show_gaze_arrows else 'OFF'}")
            elif key == ord('6'):
                show_gaze_angles = not show_gaze_angles
                print(f"[INFO] Gaze angles: {'ON' if show_gaze_angles else 'OFF'}")
            elif key == ord('7'):
                show_lookat_arrows = not show_lookat_arrows
                print(f"[INFO] Look-at arrows: {'ON' if show_lookat_arrows else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print("\n" + "="*60)
    print("FINAL DETECTION STATISTICS")
    print("="*60)
    print(f"Total frames: {total_frames}")
    print(f"\nStage 1 (Face Detection):")
    print(f"  Attempts: {stage1_attempts}")
    print(f"  Success: {stage1_success}")
    if stage1_attempts > 0:
        print(f"  Rate: {(stage1_success/stage1_attempts)*100:.1f}%")
    
    print(f"\nStage 2 (FaceMesh on cropped face):")
    print(f"  Attempts: {stage2_attempts}")
    print(f"  Success: {stage2_success}")
    if stage2_attempts > 0:
        print(f"  Rate: {(stage2_success/stage2_attempts)*100:.1f}%")
    
    print(f"\nPrediction CSV: {csv_path}")
    
    # Print validation metrics
    if metrics:
        metrics.print_summary()
        
        # Save detailed report
        metrics_path = os.path.join(SAVE_DIR, f"metrics_{session_stamp}.csv")
        metrics.save_detailed_report(metrics_path)
        print(f"\nDetailed metrics report: {metrics_path}")
    
    print("="*60)
    
    # Print performance timing summary
    perf_timer.print_summary()
    
    # Save timing data to CSV
    timing_csv_path = os.path.join(SAVE_DIR, f"timing_{session_stamp}.csv")
    perf_timer.save_to_csv(timing_csv_path)
    print(f"\nTiming data saved to: {timing_csv_path}")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()