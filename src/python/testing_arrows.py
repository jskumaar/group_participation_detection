#Draws gaze arrows for first 30 secs of test vid 1. 
#As of now, opentrack run on each cropped vid and then the data is read from csv files.
import numpy as np
import cv2
import ultralytics
from ultralytics import YOLO
import pandas as pd 
import testing_helper
import math
import static_crop
import spherical_norm

model = YOLO('yolov8s.pt')

# this is trimmed ideo of Insta360 mp4. just used for testing, only first 30 sec. 
video_path = '/Applications/trimmed_test.mp4'
cap = cv2.VideoCapture(video_path)

vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# # output ideo will b4 640 x 480 for each person
standard_size = (640, 480)

angle_threshold = 20
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
bbox_spherical = {}
gaze_vectors = {}
for person_id, file_path in saved_videos.items():
    gaze_pts_list = []
    gaze_end_pts[person_id] = gaze_pts_list
    gaze_vec_list = []
    gaze_vectors[person_id] = gaze_vec_list
    
    #people are reltatively stationary, so bounding boxes are just their location from first frame 
    (x1,y1,x2,y2) = static_crop.get_bbox(person_id)
    bbox_center_x = int((x1+x2))//2
    bbox_center_y = int((y1+y2))//2

    person_spherical_coords = testing_helper.pixel_to_spherical(bbox_center_x,bbox_center_y,vid_width,vid_height)
    lon, lat = person_spherical_coords 
    person_pos = testing_helper.spherical_to_cartesian(lon,lat)
    bbox_spherical[person_id] = (lon,lat)

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
            yaw = np.radians(yaw)
            pitch = np.radians(pitch)

            gaze_vec = testing_helper.local_to_global_gaze(yaw,pitch,lon,lat)

          
            gaze_endpoint = person_pos + gaze_vec
            gaze_endpoint /= np.linalg.norm(gaze_endpoint)  # project back to unit sphere




            gaze_lon, gaze_lat = testing_helper.cartesian_to_spherical(gaze_endpoint)
            gx, gy = testing_helper.spherical_to_pixel(gaze_lon, gaze_lat, vid_width, vid_height)
            gaze_pts_list.append((gx,gy))
            gaze_vec_list.append(gaze_vec)

            
            


#have list of gaze_pts for each person now. draw arrows and return edited video 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('demo_vid.mp4', fourcc, fps, (vid_width, vid_height))
cap = cv2.VideoCapture(video_path)  # reopen video again
frame_idx=0


font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
thickness = 2 
color = (0, 0, 255)



while True:



    arrow_length = 500
    ret, frame = cap.read()
    if not ret:
        break
    i = 0


    buffer = np.radians(15)
    for person_id, gaze_pts_list in gaze_end_pts.items():

        gaze_vector_list = gaze_vectors[person_id]
        gaze_vec = gaze_vector_list[frame_idx]


        (x1,y1,x2,y2) = static_crop.get_bbox(person_id)
        start_x = int((x1 + x2))//2
        start_y = int(y1 + 0.15 * (y2 - y1)) #about eye level. just estimation for now 
        start_point = (start_x, start_y)
        end_x, end_y = gaze_pts_list[frame_idx]
        endpt = (int(end_x), int(end_y))
        spherical_norm.draw_arrow_global(frame,start_point,endpt,(255,0,0))
       







        cv2.putText(frame, f" person: {person_id}", (start_x-200,  start_y+400), font, font_scale, color, thickness)



        #compute angle difs for each person bounding box, take smaller one as long as <30
        min_angle = np.radians(float(359))
        target = None
        for personid, pt in bbox_spherical.items():
            lon,lat = pt
            px, py = testing_helper.spherical_to_pixel(lon, lat, vid_width, vid_height)
            cv2.circle(frame,(px,py),10,(0,255,0,4))



            if personid!=person_id:
                a1,b1,a2,b2 = static_crop.get_bbox(personid)
                corners = [
                    (a1, b1),
                    (a1, b2),
                    (a2, b1),
                    (a2, b2)
                ]
                spherical_corners = [testing_helper.pixel_to_spherical(a, b, vid_width, vid_height) for a, b in corners]
                corner_vecs = [testing_helper.spherical_to_cartesian(lon, lat) for lon, lat in spherical_corners]
                bbox_center_vec = np.mean(corner_vecs, axis=0)
                bbox_center_vec /= np.linalg.norm(bbox_center_vec)  # normalize

                max_angle = max([testing_helper.angular_distance(bbox_center_vec, v) for v in corner_vecs])
                angle_to_bbox = testing_helper.angular_distance(bbox_center_vec, gaze_vec)

                
                print(f" person {person_id} angle to {personid} is {angle_to_bbox}")

                if angle_to_bbox<max_angle + buffer and angle_to_bbox<min_angle:
                    min_angle = angle_to_bbox
                    target = personid
                   






        if target is not None:
            text = f"person {person_id} looking at person {target}"
            

            # min_lon, min_lat = bbox_spherical[min_key]
            # pt_x, pt_y = testing_helper.spherical_to_pixel(min_lon, min_lat, vid_width, vid_height)
        
            cv2.putText(frame, text, (start_x-400,  start_y-400), font, font_scale, (255,0,0), thickness)

 
    out.write(frame)
    frame_idx+=1

cap.release()
out.release()
cv2.destroyAllWindows()