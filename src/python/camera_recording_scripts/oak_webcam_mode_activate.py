#!/usr/bin/env python3
"""
uvc_rgb_all.py
--------------
Automatically activates all connected DepthAI cameras as independent
UVC (USB webcam) devices. Each will appear as a standard webcam
(e.g. “OAK-D UVC”) and can be used in OBS, VLC, Zoom, etc.

Usage:
    python3 uvc_rgb_all.py
"""

import os
import time
import depthai as dai
import threading

# ==================================================
# UVC PIPELINE CREATION
# ==================================================
def get_pipeline():
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setInterleaved(False)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setFps(30)

    uvc = pipeline.createUVC()
    cam_rgb.video.link(uvc.input)

    config = dai.Device.Config()
    config.board.uvc = dai.BoardConfig.UVC(1920, 1080)
    config.board.uvc.frameType = dai.ImgFrame.Type.NV12
    pipeline.setBoardConfig(config.board)

    return pipeline


# ==================================================
# START DEVICE AS UVC CAMERA
# ==================================================
def start_uvc(device_info):
    pipeline = get_pipeline()
    try:
        with dai.Device(pipeline, device_info) as device:
            mxid = device.getMxId()
            print(f"✅ [{mxid}] UVC stream started. "
                  f"Open OBS/VLC/Camera app and select 'Luxonis' or similar.")
            while True:
                time.sleep(1)
    except Exception as e:
        print(f"Error starting UVC for {device_info.getMxId()}: {e}")


# ==================================================
# MAIN FUNCTION
# ==================================================
def main():
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        print("No DepthAI cameras detected.")
        return

    print(f"🔍 Found {len(devices)} DepthAI device(s):")
    for i, d in enumerate(devices):
        print(f" [{i}] MXID: {d.getMxId()} - {d.state.name}")

    threads = []
    for d in devices:
        t = threading.Thread(target=start_uvc, args=(d,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.5)  # slight delay to prevent USB bandwidth spike

    print("\n🎬 All connected cameras are now active as UVC webcams.")
    print("Press Ctrl+C to stop.\n")

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n Stopping all UVC streams...")
        # Threads will exit automatically when process ends.


if __name__ == "__main__":
    main()
