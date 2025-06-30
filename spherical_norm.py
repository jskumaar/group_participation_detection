import numpy as np
import math

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


def local_to_global_gaze(opentrack_yaw, opentrack_pitch, person_longitude, person_latitude):
    # Get position of the person on the unit sphere
    person_position = spherical_to_cartesian(person_longitude, person_latitude)

    # Base gaze direction is inward (i.e., toward the origin)
    base_gaze_dir = -person_position / np.linalg.norm(person_position)

    # Convert yaw/pitch to radians
    yaw_rad = math.radians(opentrack_yaw)
    pitch_rad = math.radians(opentrack_pitch)

    # Rotate the inward gaze direction in local coordinate frame
    final_gaze_dir = rotate_vector(base_gaze_dir, yaw_rad, pitch_rad)

    return tuple(final_gaze_dir)
