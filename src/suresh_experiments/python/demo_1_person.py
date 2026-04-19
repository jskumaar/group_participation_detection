#Draws gaze arrows for first 30 secs of test vid 1. 
#As of now, opentrack run on each cropped vid and then the data is read from csv files.
import numpy as np
import cv2
import spherical_norm
import static_crop

from ultralytics import YOLO
import pandas as pd 
from pyEquirectRotate.src.equirectRotate import EquirectRotate, pointRotate


model = YOLO('yolov8s.pt')


person_tracks = {}        

#stores each person's cropping pts mapped to by id. cropping points  = bbox coords 
person_loc = {}

# this is trimmed ideo of Insta360 mp4. just used for testing, only first 30 sec. 
video_path = '/Applications/demo_1_3_min.mp4'
cap = cv2.VideoCapture(video_path)
vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# # output ideo will b4 640 x 480 for each person
standard_size = (640, 480)
fps = int(cap.get(cv2.CAP_PROP_FPS))
cap.release()

#cropping vids frame by frame, then saving each vid. 
# saved_videos is a dictionary of file paths with person id as key and file path as val 


 

#create map of person_id -> file path with opentrack data 
opentrack_data_by_person = {}
opentrack_data_by_person[1] = 'person_id_1_3_min.csv'


def crop_vid(model, standard_size, video_path):
# init each persons list as empty and cropping pts as (-1,-1,-1,-1)
    i = 1
  
    person_tracks[i] = []
    person_loc[i] = (-1,-1,-1,-1)
        



    # output ideo will b4 640 x 480 for each person
    standard_size = (640, 480)
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml")

        if results[0].boxes.id is None:
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, person_id in zip(boxes, ids):

            # sometimes detecting chair as object, so only first 3 ids  = people 
            if person_id!=1:
                continue


            #cropping based off only locations on first frame, not dynamic 

            # if first frame, store cropping pts 
            if person_loc[person_id] == (-1,-1,-1,-1):
                x1, y1, x2, y2 = box.astype(int)
                person_loc[person_id] = x1, y1, x2, y2

            
            x1, y1, x2, y2 = person_loc[person_id]
            cropped = frame[y1:y2, x1:x2]
            resized_crop = cv2.resize(cropped, standard_size)
            person_tracks[person_id].append(resized_crop)

                
            

            

    cap.release()
    return person_tracks

# Save each person's video
def save_vids(fps, standard_size, person_tracks):
    video_paths = {}
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for idx, (person_id, frames) in enumerate(person_tracks.items()):
        if not frames:
            continue
        if person_id!=1:
            continue
        out = cv2.VideoWriter(f'output_person_{person_id}_3_min.mp4', fourcc, fps, standard_size)
        video_paths[person_id] = (f'static_output_person_{person_id}.mp4')
        for f in frames:
            out.write(f)
        out.release()
    return video_paths

def get_bbox(person_id):
    return person_loc[person_id]















individual_frames = crop_vid(model, standard_size, video_path)
saved_videos = save_vids(fps, standard_size, individual_frames)


#maps person_id to list of their gaze_end_pts by frame
gaze_end_pts = {}
for person_id, file_path in saved_videos.items():
    gaze_pts_list = []
    gaze_end_pts[person_id] = gaze_pts_list
    
    #people are reltatively stationary, so bounding boxes are just their location from first frame 
    (x1,y1,x2,y2) = get_bbox(person_id)
    bbox_center_x = int((x1+x2))//2
    bbox_center_y = int((y1+y2))//2

    person_spherical_coords = spherical_norm.bbox_to_spherical_coord(bbox_center_x, bbox_center_y, 
                                                                     vid_width, vid_height)
    lon, lat = person_spherical_coords 
    


    part_of_start_delay = True
    
    #reading yaw,pitch data. 

    #There is a start delay to the data. So skipping the rows of data that occur before the values
    #start changing by frame 
    df = pd.read_csv(opentrack_data_by_person[person_id])
    df = pd.read_csv(opentrack_data_by_person[person_id])
    print(df[['rawYaw', 'rawPitch', 'rawRoll']].describe())
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
    

            # get global gaze pt -> convert back to spherical -> convert back to 2d -> save pt in list
            gaze_3d = spherical_norm.local_to_global_gaze(yaw, pitch,roll, lon,lat)
            gaze_x, gaze_y, gaze_z = gaze_3d
            gaze_lon, gaze_lat = spherical_norm.cartesian_to_spherical(gaze_x, gaze_y, gaze_z)
            end_pt = spherical_norm.spherical_to_2d(gaze_lon,gaze_lat,vid_width,vid_height)
            gaze_pts_list.append(end_pt)
            # rotated_point = pointRotate(vid_height, vid_width, (bbox_center_y, bbox_center_x),
            #                              (yaw, pitch, roll))
            
            # gaze_pts_list.append(rotated_point)

#have list of gaze_pts for each person now. draw arrows and return edited video 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('demo_vid)_1_per.mp4', fourcc, fps, (vid_width, vid_height))
cap = cv2.VideoCapture(video_path)  # reopen video again
frame_idx=0
while True:
    arrow_length = 500
    ret, frame = cap.read()
    if not ret:
        break
    
    gaze_pts_list = gaze_end_pts[1]
    if frame_idx >= len(gaze_pts_list):
        continue # skip drawing for this person if we're out of gaze data
    (x1,y1,x2,y2) = get_bbox(1)
    start_x = int((x1 + x2))//2
    start_y = int(y1 + 0.15 * (y2 - y1)) #about eye level. just estimation for now 

    #i used to draw different colored arrows for each person to see easily
    
    start_point = (start_x, start_y)

        #i used to draw different colored arrows for each person to see easily
    end_x, end_y = gaze_pts_list[frame_idx]
    endpt = (int(end_x), int(end_y))
     
    spherical_norm.draw_arrow_global(frame,(start_x,start_y),endpt, (0,0,255))
        
    out.write(frame)
    frame_idx+=1

cap.release()
cv2.destroyAllWindows()








