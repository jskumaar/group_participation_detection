import numpy
import math 


# persons bounding box coords in 360 view --> spherical coordinates to solve edge issues 
def bbox_to_spherical_coord(bbox_center_x, bbox_center_y, frame_width, frame_height):
    # Normalize to [0,1]
    u = bbox_center_x / frame_width
    v = bbox_center_y / frame_height
    
    # Convert to spherical coordinates
    longitude = (u * 2 * math.pi) - math.pi  # [-π, π]
    latitude = (math.pi / 2) - (v * math.pi)  # [π/2, -π/2]
    
    return longitude, latitude


# takes in yaw,pitch,roll from opentrack & cropped feeds --> 3d vector in spherical coordinates for 360 view
def local_to_global_gaze(opentrack_yaw, opentrack_pitch, person_longitude, person_latitude):
   
    # Convert to radians
    yaw_rad = math.radians(opentrack_yaw)
    pitch_rad = math.radians(opentrack_pitch)
    
    # Add local head rotation to person's base orientation
    global_x_angle = person_longitude + yaw_rad
    global_y_angle= person_latitude + pitch_rad
    
    # Convert to 3D unit vector. Radius doesn't matter, really only direction. SO assume radius 1 
    gaze_x = math.cos(global_y_angle) * math.cos(global_x_angle)
    gaze_y = math.cos(global_y_angle) * math.sin(global_x_angle)
    gaze_z = math.sin(global_y_angle)
    
    return gaze_x, gaze_y, gaze_z


