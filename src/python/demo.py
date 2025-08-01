#Draws gaze arrows for first 30 secs of test vid 1. 
#As of now, opentrack run on each cropped vid and then the data is read from csv files.
import numpy as np
import cv2
import spherical_norm
import static_crop
import os
from ultralytics import YOLO
import pandas as pd 
from pyEquirectRotate.src.equirectRotate import EquirectRotate, pointRotate
import math


model = YOLO('yolov8s.pt')

# this is trimmed ideo of Insta360 mp4. just used for testing, only first 30 sec. 
video_path = '/Applications/trimmed_test.mp4'
cap = cv2.VideoCapture(video_path)

vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# # output ideo will b4 640 x 480 for each person
standard_size = (640, 480)
fps = int(cap.get(cv2.CAP_PROP_FPS))


#cropping vids frame by frame, then saving each vid. 
# saved_videos is a dictionary of file paths with person id as key and file path as val 
individual_frames = static_crop.crop_vid(model,cap, standard_size, video_path)
saved_videos = static_crop.save_vids(fps, standard_size, individual_frames)

 

#create map of person_id -> file path with opentrack data 
opentrack_data_by_person = {}
opentrack_data_by_person[1] = '/Users/atind/Downloads/opentrackpersonid1.csv'
opentrack_data_by_person[2] = '/Users/atind/Downloads/opentrackpersonid2.csv'
opentrack_data_by_person[3] = '/Users/atind/Downloads/opentrackpersonid3.csv'



#maps person_id to list of their gaze_end_pts by frame
gaze_end_pts = {}
for person_id, file_path in saved_videos.items():
    gaze_pts_list = []
    gaze_end_pts[person_id] = gaze_pts_list
    
    #people are reltatively stationary, so bounding boxes are just their location from first frame 
    (x1,y1,x2,y2) = static_crop.get_bbox(person_id)
    bbox_center_x = int((x1+x2))//2
    bbox_center_y = int((y1+y2))//2

    person_spherical_coords = spherical_norm.bbox_to_spherical_coord(bbox_center_x, bbox_center_y, 
                                                                     vid_width, vid_height)
    lon, lat = person_spherical_coords 
    inward_yaw = math.degrees(lon) + 180  # Flip 180°
    inward_pitch = -math.degrees(lat) 

    part_of_start_delay = True
    
    #reading yaw,pitch data. 

    #There is a start delay to the data. So skipping the rows of data that occur before the values
    #start changing by frame 
    df = pd.read_csv(opentrack_data_by_person[person_id])
    for index, row in df.iterrows():
        if part_of_start_delay == True and row['correctedTX'] == 0:
            continue 
        elif part_of_start_delay == True and row['correctedTX'] != 0:
            part_of_start_delay = False
            continue 
        else:
            yaw = row['rawYaw']
            pitch = row['rawPitch']
            roll = row['rawRoll']
            

            print(f"Input: bbox_center=({bbox_center_y}, {bbox_center_x}), rotation=({yaw}, {pitch}, {roll})")
            rotated_point = pointRotate(vid_height, vid_width, (bbox_center_y, bbox_center_x), (yaw, pitch, roll))
            print(f"Output: {rotated_point}")
            
            # # Use person's actual pixel position
            # gaze_pts_list.append(rotated_point)

            #get global gaze pt -> convert back to spherical -> convert back to 2d -> save pt in list
            gaze_3d = spherical_norm.local_to_global_gaze(yaw ,pitch,roll,lon,lat)
            gaze_x, gaze_y, gaze_z = gaze_3d
            gaze_lon, gaze_lat = spherical_norm.cartesian_to_spherical(gaze_x, gaze_y, gaze_z)
            end_pt = spherical_norm.spherical_to_2d(gaze_lon,gaze_lat,vid_width,vid_height)
            gaze_pts_list.append(end_pt)
            print("end_pt:", end_pt)


#have list of gaze_pts for each person now. draw arrows and return edited video 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('demo_vid.mp4', fourcc, fps, (vid_width, vid_height))
cap = cv2.VideoCapture(video_path)  # reopen video again
frame_idx=0
while True:
    arrow_length = 500
    ret, frame = cap.read()
    if not ret:
        break
    i = 0
    for person_id, gaze_pts_list in gaze_end_pts.items():
        (x1,y1,x2,y2) = static_crop.get_bbox(person_id)
        start_x = int((x1 + x2))//2
        start_y = int(y1 + 0.15 * (y2 - y1)) #about eye level. just estimation for now 
        start_point = (start_x, start_y)
        end_x, end_y = gaze_pts_list[frame_idx]
        endpt = (int(end_x), int(end_y))

        #i used to draw different colored arrows for each person to see easily
        spherical_norm.draw_arrow_global(frame,start_point,endpt,(255,0,0))
    #     end_point_yaw = (int(start_x + arrow_length * np.sin(np.radians(yaw))),
    #              int(start_y))

    # # Pitch arrow (vertical: down = positive pitch)
    #     end_point_pitch = (start_x,
    #                     int(start_y + arrow_length * -np.sin(np.radians(pitch))))

    #     # Roll arrow (diagonal/tilt)
    #     end_point_roll = (int(start_x + arrow_length * np.sin(np.radians(roll))),
    #                     int(start_y + arrow_length * np.cos(np.radians(roll))))

    #     # Draw arrows
    #     cv2.arrowedLine(frame, start_point, end_point_yaw, (255, 0, 0), 2)   # Blue = yaw
    #     cv2.arrowedLine(frame, start_point, end_point_pitch, (0, 255, 0), 2) # Green = pitch
    #     cv2.arrowedLine(frame, start_point, end_point_roll, (0, 0, 255), 2)  # Red = roll
                
    out.write(frame)
    frame_idx+=1

cap.release()
out.release()
cv2.destroyAllWindows()






    


