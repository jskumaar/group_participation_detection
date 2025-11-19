import cv2
import numpy as np

# === CONFIG ===
VIDEO_PATH = r"C:\Users\sures\Desktop\children_pilot_2_360_corrected_with_audio.mp4"
ARUCO_DICT = cv2.aruco.DICT_4X4_50
SHOW_RECTILINEAR_VIEW = False   # set True for 90° normal view
SAVE_OUTPUT = False
OUTPUT_PATH = "aruco_detected.mp4"

# === SETUP ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open video file.")

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Create resizable window
cv2.namedWindow("Aruco Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Aruco Detection", 1280, 640)

# Optional video writer
if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        OUTPUT_PATH, fourcc, cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(3)), int(cap.get(4)))
    )

# === FUNCTION: Equirectangular → Rectilinear (normal 90° view) ===
def equirect_to_rectilinear(img, fov_deg=90, theta=0, phi=0, out_w=800, out_h=600):
    h, w = img.shape[:2]
    fov = np.deg2rad(fov_deg)
    fx = out_w / (2 * np.tan(fov / 2))
    fy = fx

    x = np.linspace(-out_w / 2, out_w / 2, out_w)
    y = np.linspace(-out_h / 2, out_h / 2, out_h)
    xv, yv = np.meshgrid(x, y)
    zv = fx * np.ones_like(xv)

    xyz = np.stack((xv, yv, zv), axis=-1)
    # Rotation matrices (yaw θ, pitch φ)
    R_yaw = np.array([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])
    R_pitch = np.array([[1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)]])
    R = R_yaw @ R_pitch
    xyz = xyz @ R.T

    lon = np.arctan2(xyz[..., 0], xyz[..., 2])
    lat = np.arcsin(xyz[..., 1] / np.linalg.norm(xyz, axis=-1))

    u = (lon / (2 * np.pi) + 0.5) * w
    v = (0.5 - lat / np.pi) * h
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)

    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

# === PROCESS FRAMES ===
print("Processing video... Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(frame)

    # Draw detections
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i, corner in zip(ids, corners):
            center = corner[0].mean(axis=0).astype(int)
            cv2.putText(frame, f"ID: {int(i)}", (center[0] - 20, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # === DISPLAY ===
    if SHOW_RECTILINEAR_VIEW:
        # Show 90° normal view facing forward (adjust theta for left/right)
        view = equirect_to_rectilinear(frame, fov_deg=90, theta=0, phi=0)
    else:
        # Show full equirectangular frame (resized to fit screen)
        display_w = 1280
        display_h = int(frame.shape[0] * display_w / frame.shape[1])
        view = cv2.resize(frame, (display_w, display_h))

    cv2.imshow("Aruco Detection", view)

    if SAVE_OUTPUT:
        out.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        SHOW_RECTILINEAR_VIEW = not SHOW_RECTILINEAR_VIEW  # toggle mode

cap.release()
if SAVE_OUTPUT:
    out.release()
cv2.destroyAllWindows()
print("Done.")
