import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os, threading
import time

# Configuration
image_filename = 'image.jpg'
model_path = 'pose_landmarker.task'
output_folder = 'output_results'
time_log_file = os.path.join(output_folder, 'time_logs.json')
num_iterations = 100  # Number of times each thread processes the image
num_threads = 3  # Number of threads

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

program_execution_start_time = time.time()


# Define MediaPipe options
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    output_segmentation_masks=True,
    running_mode=VisionRunningMode.IMAGE)

time_logs = []

# Functions
def process_image(detector, image_path):
    """Process an image with the given MediaPipe detector."""
    start_time = time.time()
    mp_image = mp.Image.create_from_file(image_path)
    load_time = time.time() - start_time

    start_time = time.time()
    result = detector.detect(mp_image)
    process_time = time.time() - start_time
    
    return result, mp_image, load_time, process_time



def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image



def worker(thread_id):
    """Thread worker function to process images."""
    detector = vision.PoseLandmarker.create_from_options(options)
    
    for i in range(num_iterations):
        detection_result, mp_image, load_time, process_time = process_image(detector, image_filename)
        
        # Draw landmarks
        start_time = time.time()
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        draw_time = time.time() - start_time
        
        start_time = time.time()
        image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        # Save the annotated image
        image_path = os.path.join(output_folder, f'thread_{thread_id}_iter_{i}.png')
        cv2.imwrite(image_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        save_time = time.time() - start_time

        # Extract and save landmarks
        frame_results = []
        if detection_result.pose_landmarks:
            for landmark in detection_result.pose_landmarks[0]:
                frame_results.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
        json_path = os.path.join(output_folder, f'thread_{thread_id}_iter_{i}.json')
        with open(json_path, 'w') as f:
            json.dump(frame_results, f)

        log_entry = {
            'thread_id': thread_id,
            'iteration': i,
            'load_time': load_time,
            'process_time': process_time,
            'draw_time': draw_time,
            'save_time': save_time
        }
        time_logs.append(log_entry)
        
        print(f'Thread {thread_id} - Iteration {i} completed. Load Time: {load_time:.4f}s, Process Time: {process_time:.4f}s, Draw Time: {draw_time:.4f}s, Save Time: {save_time:.4f}s')



# Create and start threads
threads = []
for t in range(num_threads):
    thread = threading.Thread(target=worker, args=(t,))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

# Save time logs to a file
with open(time_log_file, 'w') as f:
    json.dump(time_logs, f, indent=4)


total_execution_time = time.time() - program_execution_start_time

print("All threads have completed execution. It took {:.4f}s".format(total_execution_time))
##################################



# # with PoseLandmarker.create_from_options(options) as landmarker:
# # The landmarker is initialized. Use it here.
# # ...

# # Load the input image from an image file.
# image_filename = 'image.jpg'
# mp_image = mp.Image.create_from_file(image_filename)

# # # Convert BGR to RGB (matplotlib expects images in RGB format)
# # image_rgb = cv2.cvtColor(mp_image, cv2.COLOR_BGR2RGB)


# pose_landmarker_result = process_image(image_filename)

# annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)

# # Convert BGR to RGB (matplotlib expects images in RGB format)
# image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# # Display the image using matplotlib
# plt.imshow(image_rgb)
# plt.axis('off')  # Turn off the axis
# plt.show()



# # Extract pose landmarks
# if pose_landmarker_result.pose_landmarks:
#   frame_results = []
#   for landmark in pose_landmarker_result.pose_landmarks[0]:
#     frame_results.append({
#         'x': landmark.x,
#         'y': landmark.y,
#         'z': landmark.z,
#         'visibility': landmark.visibility
#     })

# with open('pose_results_f{image_filename}.json', 'w') as f:
#   json.dump(frame_results, f)
