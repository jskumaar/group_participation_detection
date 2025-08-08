import numpy as np
import cv2

# === HELPERS ===

def pixel_to_spherical(x, y, W, H):
    lon = (x / W) * 2 * np.pi - np.pi         # [-π, π]
    lat = np.pi/2 - (y / H) * np.pi           # [π/2, -π/2]
    return lon, lat

def spherical_to_cartesian(lon, lat, radius=1.0):
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return np.array([x, y, z])

def cartesian_to_spherical(vec):
    x, y, z = vec
    lon = np.arctan2(y, x)
    lat = np.arcsin(z / np.linalg.norm(vec))
    return lon, lat

def spherical_to_pixel(lon, lat, W, H):
    x = int((lon + np.pi) / (2 * np.pi) * W)
    y = int((np.pi/2 - lat) / np.pi * H)
    return x, y

def gaze_vector_from_yaw_pitch(yaw_rad, pitch_rad):
    # Standard camera frame: z = forward, x = right, y = up
    x = np.sin(yaw_rad) * np.cos(pitch_rad)
    y = np.sin(pitch_rad)
    z = np.cos(yaw_rad) * np.cos(pitch_rad)
    return np.array([x, y, z])



def local_to_global_gaze(opentrack_yaw, opentrack_pitch, person_longitude, person_latitude):
    """
    Convert local yaw/pitch (relative to person's forward) into a global gaze direction vector.
    All angles in radians.
    Coordinate system:
        X right
        Y up
        Z forward (from camera’s POV)
    """
    # 1. Person's position on sphere (radius = 1)
    person_pos = spherical_to_cartesian(person_longitude, person_latitude)

    # 2. Person’s forward vector (looking toward sphere center)
    forward = -person_pos / np.linalg.norm(person_pos)

    # 3. World "up" vector (global Y)
    world_up = np.array([0, 1, 0])

    # 4. Person’s right vector
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)

    # 5. Person’s true up vector
    up = np.cross(forward, right)

    # 6. Rotation matrix from local to world
    R = np.column_stack((right, up, forward))  # local X, local Y, local Z

    # 7. Local gaze vector from yaw/pitch
    local_gaze = gaze_vector_from_yaw_pitch(opentrack_yaw, opentrack_pitch)

    # 8. Transform to world coordinates
    global_gaze = R @ local_gaze
    return global_gaze / np.linalg.norm(global_gaze)


def angular_distance(vec1, vec2):
    dot = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
    return np.arccos(dot)  # radians

