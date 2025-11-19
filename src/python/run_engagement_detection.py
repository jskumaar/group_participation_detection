# #!/usr/bin/env python3
# import cv2, numpy as np, math, os, time
# from datetime import datetime
# from ultralytics import YOLO
# import mediapipe as mp
# from collections import deque

# # =========================
# # CONFIG
# # =========================
# USE_CAMERA = False
# VIDEO_FILE_PATH = r"G:\My Drive\Research\(Shared) Project_mixed_group_interaction\Pilot_data_collection\Session_1\Pilot_1_360_hd_interaction.mp4"

# STEP_MODE = True     # True = space to step frame-by-frame, False = continuous playback

# COLORS = [(0,0,255), (0,255,0), (255,0,0), (0,255,255)]

# CAMERA_INDEX        = 1
# CAM_WIDTH           = 2880
# CAM_HEIGHT          = 1440

# YOLO_MODEL_PATH     = "yolov8n.pt"
# YOLO_CONF_THRESHOLD = 0.35
# YOLO_IMGSZ          = 960

# GNOMONIC_OUT_W      = 512
# GNOMONIC_OUT_H      = 512
# GNOMONIC_ROLL_DEG   = 180.0

# SAVE_DIR            = "live_out"
# SHOW_EQUIRECT_WINDOW = True
# EQUIRECT_DISPLAY_MAX_W = 1280

# SHOW_FACE_BBOX       = True
# SHOW_PERSON_BBOX     = True

# # =========================
# # FaceMesh + PnP
# # =========================
# FACE_3D_IDXS = [1, 152, 33, 263, 61, 291]
# FACE_3D_MODEL = np.array([
#     (0.0, 0.0, 0.0),
#     (0.0, -63.6, -12.5),
#     (-43.3, 32.7, -26.0),
#     (43.3, 32.7, -26.0),
#     (-28.9, -28.9, -24.1),
#     (28.9, -28.9, -24.1)
# ], dtype=np.float64)

# mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
#     max_num_faces=1, refine_landmarks=True,
#     min_detection_confidence=0.5, min_tracking_confidence=0.5
# )

# GAZE_HISTORY = deque(maxlen=30)

# # ---------- math helpers ----------
# def _unit(v):
#     v = np.asarray(v, dtype=np.float64)
#     n = np.linalg.norm(v)
#     return v / (n + 1e-9)

# def _dir_from_lonlat(lam, phi):
#     c = np.cos(phi)
#     return np.array([c*np.cos(lam), np.sin(phi), c*np.sin(lam)], dtype=np.float64)

# def _basis_from_forward(f):
#     f = f/np.linalg.norm(f)
#     up = np.array([0, 1, 0.0])
#     u = up - f*np.dot(up, f)
#     if np.linalg.norm(u) < 1e-9:
#         up = np.array([1, 0, 0.0])
#         u = up - f*np.dot(up, f)
#     u = u/np.linalg.norm(u)
#     r = np.cross(u, f)
#     r = r/np.linalg.norm(r)
#     return r, u, f

# def _lonlat_from_px(u, v, W, H):
#     lam = 2*math.pi*(u/W) - math.pi
#     phi = math.pi/2 - math.pi*(v/H)
#     return lam, phi

# def dir_to_equirect_px(D, W, H):
#     Dx, Dy, Dz = D
#     lam = math.atan2(Dz, Dx)
#     phi = math.asin(np.clip(Dy, -1.0, 1.0))
#     u = (lam + math.pi) / (2 * math.pi) * W
#     v = (math.pi/2 - phi) / math.pi * H
#     return int(round(u)), int(round(v))

# def crop_px_to_global_dir(px, py, crop_w, crop_h, fov_deg, r_vec, u_vec, f_vec):
#     x_ndc = (px / crop_w - 0.5) * 2
#     y_ndc = (py / crop_h - 0.5) * 2
#     hfov = math.radians(fov_deg)
#     vfov = 2 * math.atan(math.tan(hfov/2) * (crop_h/crop_w))
#     lx = math.tan(hfov/2) * x_ndc
#     ly = math.tan(vfov/2) * y_ndc
#     lz = 1.0
#     dir_cam = np.array([lx, ly, lz])
#     dir_cam /= np.linalg.norm(dir_cam)
#     return _unit(r_vec*dir_cam[0] + u_vec*dir_cam[1] + f_vec*dir_cam[2])

# # ---------- pose estimation ----------
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
#     face_2d = []
#     for idx in FACE_3D_IDXS:
#         lm = lms.landmark[idx]
#         face_2d.append((lm.x * w, lm.y * h))
#     face_2d = np.array(face_2d, dtype=np.float64)
#     cam_matrix = get_camera_matrix(w, h)
#     dist = np.zeros((4, 1), dtype=np.float64)
#     success, rvec, tvec = cv2.solvePnP(FACE_3D_MODEL, face_2d,
#                                        cam_matrix, dist,
#                                        flags=cv2.SOLVEPNP_ITERATIVE)
#     if not success:
#         return None
#     return head_box, rvec, lms

# # ---------- detection ----------
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
#                          key=lambda d: (d["bbox"][2]-d["bbox"][0]) *
#                                        (d["bbox"][3]-d["bbox"][1]),
#                          reverse=True)
#     return dets_sorted[:3]

# # ---------- main ----------
# def main():
#     os.makedirs(SAVE_DIR, exist_ok=True)

#     yolo = load_yolo(YOLO_MODEL_PATH)
#     cap = cv2.VideoCapture(VIDEO_FILE_PATH if not USE_CAMERA else CAMERA_INDEX)
#     if USE_CAMERA:
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

#     frame_idx = 0
#     current_dets = []

#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break
#         H, W = frame.shape[:2]
#         frame_idx += 1
#         equi_with_pose = frame.copy()

#         # Person detection
#         current_dets = detect_persons(frame, yolo,
#                                       conf=YOLO_CONF_THRESHOLD,
#                                       imgsz=YOLO_IMGSZ)
#         current_dets = select_top3_by_size(current_dets)
#         for pid, d in enumerate(current_dets, start=1):
#             d["id"] = f"P{pid}"
#             d["color"] = COLORS[(pid-1) % len(COLORS)]

#         # Draw person bboxes
#         if SHOW_PERSON_BBOX:
#             for d in current_dets:
#                 x1, y1, x2, y2 = d["bbox"]
#                 cv2.rectangle(equi_with_pose, (x1, y1), (x2, y2), d["color"], 2)
#                 cv2.putText(equi_with_pose, d["id"], (x1, max(20, y1-10)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, d["color"], 2)

#         all_head_poses = []
#         for d in current_dets:
#             x1, y1, x2, y2 = d["bbox"]
#             crop = frame[y1:y2, x1:x2]
#             pose = estimate_head_pose_with_visuals(crop)

#             if not pose:
#                 all_head_poses.append(None)
#                 continue

#             head_box, rvec, lms = pose

#             # --- spherical math ---
#             rmat, _ = cv2.Rodrigues(rvec)
#             f_vec_crop = rmat @ np.array([0, 0, 1.0])
#             f_vec_crop = _unit(f_vec_crop)

#             # Basis at this bbox center
#             u, v = d["center"]
#             lam, phi = _lonlat_from_px(u, v, W, H)
#             r_basis, u_basis, f_basis = _basis_from_forward(_dir_from_lonlat(lam, phi))

#             f_vec_global = (r_basis * f_vec_crop[0] +
#                             u_basis * f_vec_crop[1] +
#                             f_basis * f_vec_crop[2])
#             f_vec_global = _unit(f_vec_global)
#             all_head_poses.append(f_vec_global)

#             # --- visualization in equirect ---
#             # Nose landmark → global coords
#             nose_lm = lms.landmark[1]
#             nose_x = nose_lm.x * crop.shape[1]
#             nose_y = nose_lm.y * crop.shape[0]
#             nose_dir = crop_px_to_global_dir(nose_x, nose_y,
#                                              crop.shape[1], crop.shape[0],
#                                              60,  # approx FOV
#                                              r_basis, u_basis, f_basis)
#             nose_px = dir_to_equirect_px(nose_dir, W, H)

#             arrow_end = dir_to_equirect_px(f_vec_global, W, H)

#             # Face bbox projected
#             if SHOW_FACE_BBOX:
#                 corners_crop = [(head_box[0], head_box[1]),
#                                 (head_box[2], head_box[1]),
#                                 (head_box[2], head_box[3]),
#                                 (head_box[0], head_box[3])]
#                 corners_eq = [dir_to_equirect_px(
#                                 crop_px_to_global_dir(cx, cy,
#                                                       crop.shape[1], crop.shape[0],
#                                                       60,
#                                                       r_basis, u_basis, f_basis),
#                                 W, H)
#                               for cx, cy in corners_crop]
#                 cv2.polylines(equi_with_pose, [np.array(corners_eq, np.int32)],
#                               isClosed=True, color=d["color"], thickness=2)

#             # Nose + arrow
#             cv2.circle(equi_with_pose, nose_px, 4, d["color"], -1)
#             cv2.arrowedLine(equi_with_pose, nose_px, arrow_end,
#                             d["color"], 2, tipLength=0.25)

#         # Overlay frame info
#         disp_w = min(EQUIRECT_DISPLAY_MAX_W, W)
#         disp_h = int(disp_w / (W / float(H)))
#         equi_disp = cv2.resize(equi_with_pose, (disp_w, disp_h))
#         cv2.putText(equi_disp, f"Frame: {frame_idx}", (50, disp_h-50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

#         cv2.imshow("Equirect Preview", equi_disp)
#         if STEP_MODE:
#             key = cv2.waitKey(0) & 0xFF
#             if key == ord('q'): break
#         else:
#             if cv2.waitKey(1) & 0xFF == ord('q'): break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
