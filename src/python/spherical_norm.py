import numpy as np
import math
import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np

def bbox_to_spherical_coord(bbox_center_x, bbox_center_y, frame_width, frame_height):
    u = bbox_center_x / frame_width
    v = bbox_center_y / frame_height
    
    longitude = (u * 2 * math.pi) - math.pi  # [-π, π]
    latitude = (math.pi / 2) - (v * math.pi)  # [π/2, -π/2]
    
    return longitude, latitude


def spherical_to_cartesian(longitude, latitude, radius=1.0):
    x = radius * math.cos(latitude) * math.cos(longitude)
    y = radius * math.cos(latitude) * math.sin(longitude)
    z = radius * math.sin(latitude)
    return np.array([x, y, z])


def rotate_vector(vector, yaw, pitch):
    """
    Rotates a vector by yaw (left/right) and pitch (up/down).
    This assumes:
      - yaw is a rotation around the Z-axis (vertical)
      - pitch is a rotation around the local X-axis (side-to-side)
    """
    # Rotation matrix for yaw around Z
    R_yaw = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Rotation matrix for pitch around X
    R_pitch = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch),  math.cos(pitch)]
    ])
    
    # Combine rotations: pitch then yaw
    R = R_yaw @ R_pitch
    
    return R @ vector




import numpy as np

def euler_to_rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    # Convert degrees to radians
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)

    # Rotation around X-axis (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    # Rotation around Y-axis (pitch)
    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # Rotation around Z-axis (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix: R = Rz * Ry * Rx (yaw → pitch → roll)
    R = Rz @ Ry @ Rx
    return R









def local_to_global_gaze(opentrack_yaw, opentrack_pitch, opentrack_roll, person_longitude, person_latitude):
    # Get position of the person on the unit sphere


    person_position = spherical_to_cartesian(person_longitude, person_latitude,radius = 1)

    # Base gaze direction is inward (i.e., toward the origin)
    base_gaze_dir = -person_position / np.linalg.norm(person_position)
    

    # Convert yaw/pitch to radians
    yaw_rad = math.radians(opentrack_yaw)
    pitch_rad = math.radians(opentrack_pitch)

    # Rotate the inward gaze direction in local coordinate frame
    rotated = euler_to_rotation_matrix(opentrack_yaw, opentrack_pitch, opentrack_roll)

    

    final_gaze = rotated @ base_gaze_dir
    return tuple(final_gaze)
    # Step 1: Person's position on unit sphere
    # person_position = spherical_to_cartesian(person_longitude, person_latitude, radius=1)

    # # Step 2: Gaze vector is inward
    # base_gaze = -person_position / np.linalg.norm(person_position)

    # # Step 3: Construct rotation using yaw, pitch, roll (in degrees)
    # r = R.from_euler('ZYX', [opentrack_yaw, opentrack_pitch, opentrack_roll], degrees=True)

    # # Step 4: Apply local rotation to the base gaze direction
    # rotated_gaze = r.apply(base_gaze)

    # # Step 5: Convert to spherical → equirectangular image (2D)

    # return tuple(rotated_gaze)

   
   


#final_gaze_dir is a tuple 
#image is the global image to draw on 
def draw_arrow_global(image, start_pt, final_gaze_dir, color):
    cv2.arrowedLine(image,start_pt, final_gaze_dir, color, thickness = 2)

def cartesian_to_spherical(x, y, z):
    radius = math.sqrt(x**2 + y**2 + z**2)
    longitude = math.atan2(z,x)  # range: [-π, π]
    latitude = math.asin(y / radius)  # range: [-π/2, π/2]
    return longitude, latitude

def spherical_to_2d(longitude, latitude, frame_width, frame_height):
    # Clamp longitude to [-π, π]
    longitude = max(-math.pi, min(math.pi, longitude))
    
    # Clamp latitude to [-π/2, π/2]
    latitude = max(-math.pi/2, min(math.pi/2, latitude))
    
    # Convert to normalized coordinates [0, 1]
    u = (longitude + math.pi) / (2 * math.pi)
    v = (math.pi / 2 - latitude) / math.pi
    
    # Convert to pixel coordinates with bounds checking
    x = max(0, min(frame_width - 1, round(u * frame_width)))
    y = max(0, min(frame_height - 1, round(v * frame_height)))
    
    return x, y

