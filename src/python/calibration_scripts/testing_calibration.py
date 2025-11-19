# import cv2
# import cv2.aruco as aruco
# import numpy as np

# # # Load predefined dictionary (e.g., 6x6_250)
# # aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
# # det_params = aruco.DetectorParameters()
# # det_params.adaptiveThreshWinSizeMin = 3
# # det_params.adaptiveThreshWinSizeMax = 45
# # det_params.adaptiveThreshWinSizeStep = 10
# # det_params.minMarkerPerimeterRate = 0.01
# # det_params.maxMarkerPerimeterRate = 4.0
# # det_params.polygonalApproxAccuracyRate = 0.03
# # det_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX


# # Define dictionary and detector parameters
# # aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
# aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
# det_params = aruco.DetectorParameters()
# detector = aruco.ArucoDetector(aruco_dict, det_params)

# # Define GridBoard (2x3 markers, marker size=0.1145m, separation=0.023m)
# board = aruco.GridBoard(
#     (2, 3),    # number of rows, columns
#     0.1145,  # marker length in meters
#     0.023,   # marker separation in meters
#     cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
# )


# # Storage for calibration
# all_corners = []
# all_ids = []
# image_size = None

# cap = cv2.VideoCapture(0)
# print("Press SPACE to capture a frame, 'c' to calibrate, ESC to quit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     image_size = gray.shape[::-1]

#     # Detect markers
#     corners, ids, rejected = detector.detectMarkers(gray)


#     if ids is not None and len(ids) > 0:
#         # Draw detected markers (green borders with IDs)
#         aruco.drawDetectedMarkers(frame, corners, ids)
#         cv2.putText(frame, f"{len(ids)} markers detected",
#                     (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
#                     1.0, (0, 255, 0), 2, cv2.LINE_AA)
#     else:
#         cv2.putText(frame, "No markers detected",
#                     (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
#                     1.0, (0, 0, 255), 2, cv2.LINE_AA)

#     cv2.imshow("Calibration", frame)
#     key = cv2.waitKey(1) & 0xFF

#     if key == 27:  # ESC
#         break
#     elif key == 32:  # SPACE
#         if ids is not None and len(ids) > 0:

#             all_corners.append(corners)   # force to list
#             all_ids.append(ids.copy())            

#             print(f"✅ Captured frame with {len(ids)} markers")

#         else:
#             print("⚠️ No markers detected in this frame!")
#     elif key == ord('c'):  # Run calibration
        
#         # Debug
#         for i,(c,iid) in enumerate(zip(all_corners, all_ids)):
#             print(f"Frame {i}: corners type={type(c)}, len={len(c)}, "
#                 f"first shape={c[0].shape}, ids shape={iid.shape}, ids dtype={iid.dtype}")


#         if len(all_corners) > 0:
            
#             try:
#                 counter = np.array([len(ids) for ids in all_ids], dtype=np.int32)

#                 print('Corners:', all_corners)
#                 print('IDs:', all_ids)
#                 print('Counter:', counter)

#                 print("Running calibration with", len(all_corners), "frames...")


#                 ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(
#                     all_corners,
#                     all_ids,
#                     counter,  # counter
#                     board,
#                     image_size,
#                     None,
#                     None
#                 )

#                 print("Reprojection error:", ret)
#                 print("Camera matrix:\n", mtx)
#                 print("Distortion coefficients:\n", dist)

#                 # Save calibration
#                 np.savez("calibration_data.npz", mtx=mtx, dist=dist)
#                 print("Calibration saved to calibration_data.npz")


#             except Exception as e:
#                 print(f"❌ Calibration failed with calibrateCameraAruco: {e}")
                
#                 # Method 2: Fallback to regular calibrateCamera
#                 print("Trying fallback method with calibrateCamera...")
#                 try:
#                     # Convert ArUco data to standard calibration format
#                     object_points = []
#                     image_points = []
                    
#                     for corners, ids in zip(all_corners, all_ids):
#                         if len(corners) > 0:
#                             # Get object points for detected markers
#                             obj_pts, img_pts = board.matchImagePoints(corners, ids)
#                             if obj_pts is not None and len(obj_pts) > 0:
#                                 object_points.append(obj_pts)
#                                 image_points.append(img_pts)
                    
#                     if len(object_points) > 0:
#                         ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
#                             object_points, image_points, image_size, None, None
#                         )
                        
#                         print("✅ Fallback calibration successful!")
#                         print("Reprojection error:", ret)
#                         print("Camera matrix:\n", mtx)
#                         print("Distortion coefficients:\n", dist)
                        
#                         # Save calibration
#                         np.savez("calibration_data.npz", mtx=mtx, dist=dist, ret=ret)
#                         print("Calibration saved to calibration_data.npz")
#                     else:
#                         print("❌ No valid object points found for calibration")
                        
#                 except Exception as e2:
#                     print(f"❌ Fallback calibration also failed: {e2}")
#         else:
#             print("⚠️ No frames collected for calibration!")

# cap.release()
# cv2.destroyAllWindows()


###################################################

import cv2
import cv2.aruco as aruco
import numpy as np
import math
from pathlib import Path

# -------- Helpers --------
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
    """Project equirectangular panorama into a perspective view (gnomonic)"""
    H, W = equi.shape[:2]
    r, u, f = build_camera_basis(center_dir)
    fov = math.radians(fov_deg)
    fx = fy = 0.5 * out_w / math.tan(0.5 * fov)
    cx, cy = out_w * 0.5, out_h * 0.5

    ys, xs = np.indices((out_h, out_w), dtype=np.float32)
    x_cam = (xs - cx) / fx
    y_cam = (ys - cy) / fy
    z_cam = np.ones_like(x_cam)

    norm = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
    x_cam /= norm; y_cam /= norm; z_cam /= norm

    dir_world = (x_cam[...,None]*r + y_cam[...,None]*u + z_cam[...,None]*f)

    xw, yw, zw = dir_world[...,0], dir_world[...,1], dir_world[...,2]
    lon = np.arctan2(zw, xw)
    lat = np.arcsin(yw)
    u_e = (lon + math.pi) / (2.0*math.pi) * W
    v_e = (math.pi/2.0 - lat) / math.pi * H

    map_x = u_e.astype(np.float32)
    map_y = v_e.astype(np.float32)
    # persp = cv2.remap(equi, map_x, map_y, interpolation=cv2.INTER_LINEAR,
    #                   borderMode=cv2.BORDER_WRAP)
    persp = cv2.remap(equi, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                  borderMode=cv2.BORDER_WRAP)

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    return persp, K

# -------- Main test --------
def main():
    # Load one of your saved equirect images
    img_path = "calib_data\calib_imgs_20250824_172924/frame_000030.jpg"   # change this
    img_name = Path(img_path).stem
    equi = cv2.imread(img_path)
    if equi is None:
        raise RuntimeError(f"Could not read {img_path}")

    # Nadir direction (straight down) from camera center
    center_dir = np.array([0, -1, 0], np.float32)   # y-down
    persp, K = gnomonic_sample(equi, center_dir, fov_deg=80, out_w=2560, out_h=2560)


    # --- Detect markers on perspective ---
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)  # match your board!
    params = aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 45
    params.adaptiveThreshWinSizeStep = 10
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = aruco.ArucoDetector(dictionary, params)

    gray = cv2.cvtColor(persp, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    debug = persp.copy()
    if ids is not None and len(ids) > 0:
        aruco.drawDetectedMarkers(debug, corners, ids)
        print(f"✅ Detected {len(ids)} markers: {ids.ravel()}")
    else:
        print("❌ No markers detected")

    # cv2.imshow("Perspective (nadir)", debug)
    cv2.imwrite(f"perspective_{img_name}.jpg", debug)
    cv2.imwrite(f"debug_gray_{img_name}_gray.jpg", gray)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
