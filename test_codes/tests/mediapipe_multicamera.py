import depthai as dai
import mediapipe as mp
import cv2
import threading
import signal
import sys
import os
import csv
from datetime import datetime
import time
import psutil  # For resource monitoring

# Create output directories for saving data
base_output_dir = "../../pose_results"
os.makedirs(base_output_dir, exist_ok=True)

# Event to signal threads to stop
stop_event = threading.Event()

# Resource monitoring data
resource_data = {}

# Function to monitor system resources and print every 5 seconds
def monitor_resources():
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        cpu_percent = psutil.cpu_percent(interval=1)
        mem_info = process.memory_info()
        memory_mb = mem_info.rss / (1024 * 1024)

        # Aggregate resource usage
        resource_data['cpu_percent'] = cpu_percent
        resource_data['memory_mb'] = memory_mb

        # Print resource usage and FPS every 5 seconds
        time.sleep(5)
        print("\n--- Resource Usage ---")
        print(f"CPU Usage: {cpu_percent}%")
        print(f"Memory Usage: {memory_mb:.2f} MB")
        for camera_name, stats in resource_data.get('fps', {}).items():
            fps = stats['frames'] / stats['elapsed_time'] if stats['elapsed_time'] > 0 else 0
            print(f"{camera_name} FPS: {fps:.2f}")

# Function to initialize and process each camera
def process_camera(device_id, window_name):
    # Initialize MediaPipe Pose separately for each camera
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,  # Better for video streams
        model_complexity=1,       # Adjust for performance
        smooth_landmarks=True,    # Smooth out jitter
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Create output directories for this camera
    camera_output_dir = os.path.join(base_output_dir, window_name)
    images_dir = os.path.join(camera_output_dir, "images")
    videos_dir = os.path.join(camera_output_dir, "videos")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    # Create pipeline for DepthAI
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)

    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    device_info = dai.DeviceInfo(device_id)

    # Prepare CSV file for saving pose results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(camera_output_dir, f"{window_name}_pose_{timestamp}.csv")

    # Prepare Video Writer
    video_filename = os.path.join(videos_dir, f"{window_name}_video_{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (640, 480))

    with open(csv_filename, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["Frame", "Landmark", "X", "Y", "Z", "Visibility"])

        with dai.Device(pipeline, device_info, maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS) as device:
            video_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            frame_count = 0
            start_time = time.time()

            resource_data.setdefault('fps', {})
            resource_data['fps'][window_name] = {'frames': 0, 'start_time': start_time, 'elapsed_time': 0}

            while not stop_event.is_set():
                in_frame = video_queue.tryGet()
                if in_frame is None:
                    continue

                frame = in_frame.getCvFrame()
                frame_count += 1

                # Convert BGR to RGB for MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Perform Pose Detection with this camera's MediaPipe instance
                results = pose.process(image_rgb)

                # Save pose landmarks if detected
                if results.pose_landmarks:
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        csv_writer.writerow([
                            frame_count,
                            idx,
                            landmark.x,
                            landmark.y,
                            landmark.z,
                            landmark.visibility
                        ])

                    # Draw Pose landmarks on the frame
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Write the frame to the video
                video_writer.write(frame)

                # Update FPS data
                current_time = time.time()
                elapsed_time = current_time - start_time
                resource_data['fps'][window_name]['frames'] = frame_count
                resource_data['fps'][window_name]['elapsed_time'] = elapsed_time

                # Display the result
                cv2.imshow(window_name, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break

    # Release resources for this camera
    pose.close()
    video_writer.release()
    cv2.destroyWindow(window_name)
    print(f"Pose results saved to: {csv_filename}")
    print(f"Video saved to: {video_filename}")

# Signal handler to catch Ctrl+C and close gracefully
def signal_handler(sig, frame):
    print("\nGracefully shutting down...")
    stop_event.set()
    cv2.destroyAllWindows()
    sys.exit(0)

# Register the signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# Get all available DepthAI devices
devices = dai.Device.getAllAvailableDevices()

if not devices:
    print("No DepthAI devices found!")
    sys.exit(0)

# Extract MX IDs of connected devices
camera_ids = [device.getMxId() for device in devices]

# Create and start threads for each camera
threads = []
for idx, cam_id in enumerate(camera_ids):
    print('Connecting camera with device id:', cam_id)
    thread = threading.Thread(target=process_camera, args=(cam_id, f"Camera_{idx+1}"))
    thread.start()
    threads.append(thread)

# Start resource monitoring thread
monitor_thread = threading.Thread(target=monitor_resources)
monitor_thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

# Stop resource monitoring
stop_event.set()
monitor_thread.join()

print("All threads have been closed gracefully.")
