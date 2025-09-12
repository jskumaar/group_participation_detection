import cv2
from ultralytics import YOLO
import numpy as np
import os


model = YOLO('yolov8s.pt')



# this is trimmed ideo of Insta360 mp4. just used for testing, only first 30 sec. 
video_path = 'trimmed_test.mp4'
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))



#maps each persons_id to a list of cropped images with each image = a view of that person in a single frame. 
#can stitch this images together to create a singled view of that person 
person_tracks = {}        



i = 1
while i <4:
    person_tracks[i] = []
    i+=1




# output ideo will b4 640 x 480 for each person
standard_size = (640, 480)

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
        if person_id > 3:
            continue


        #cropping dynamically based on person's location on each frame 
        
        x1, y1, x2, y2 = box.astype(int)
        cropped = frame[y1:y2, x1:x2]
        resized_crop = cv2.resize(cropped, standard_size)
        person_tracks[person_id].append(resized_crop)

            
        

        

cap.release()

# Save each person's video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
for idx, (person_id, frames) in enumerate(person_tracks.items()):
    if not frames:
        continue
    out = cv2.VideoWriter(f'dynamic_output_person_{idx}.mp4', fourcc, fps, standard_size)
    for f in frames:
        out.write(f)
    out.release()
