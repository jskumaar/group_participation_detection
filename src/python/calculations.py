import numpy as np

def get_look_direction(yaw, pitch, roll, x, y, z):

    # Convert angles to radians if they're in degrees
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    
    # Create rotation matrices
    # Rotation around Y axis (yaw)
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    
    # Rotation around X axis (pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    
    # Rotation around Z axis (roll)
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    
    # Initial forward direction (assuming looking along positive Z axis)
    forward = np.array([0, 0, 1])
    
    # Apply rotations in order: roll -> pitch -> yaw
    direction = Ry @ Rx @ Rz @ forward
    
    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)
    
    # Create the origin point
    origin = np.array([x, y, z])
    
    return direction, origin

def calculate_perpendicular_distance(direction_vector, origin_point, target_point):
    # Convert inputs to numpy arrays if they aren't already
    direction_vector = np.array(direction_vector)
    origin_point = np.array(origin_point)
    target_point = np.array(target_point)
    
    # Calculate vector from origin to target point
    origin_to_target = target_point - origin_point
    
    # Calculate the projection of origin_to_target onto the direction vector
    projection = np.dot(origin_to_target, direction_vector)
    
    # Calculate the closest point on the line to the target point
    closest_point = origin_point + projection * direction_vector
    
    # Calculate the perpendicular distance
    distance = np.linalg.norm(target_point - closest_point)
    
    return distance



