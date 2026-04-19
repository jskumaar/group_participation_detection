# 📷 Insta360 SDK Setup on Windows

This guide provides steps to install and use the Insta360 SDK on Windows for camera control and real-time stitching.

---

## 🔧 Prerequisites

- Windows 10 or 11
- Visual Studio 2022 (with Desktop C++ development)
- CMake (added to PATH)
- OpenCV (e.g., 3.4.10 or higher)

---

## 📦 SDK Repositories

| SDK | Purpose | Link |
|-----|---------|------|
| CameraSDK | Control Insta360 camera over USB | [Desktop-CameraSDK-Cpp](https://github.com/Insta360Develop/Desktop-CameraSDK-Cpp) |
| MediaSDK | Real-time stitching and video decoding | [Desktop-MediaSDK-Cpp](https://github.com/Insta360Develop/Desktop-MediaSDK-Cpp) |

---

## 🪟 Installing Insta360 SDK on Windows

### 1. 🔌 USB Driver Setup

To allow the SDK to communicate with the Insta360 camera over USB:

- **Install the USB driver**: Follow the steps in this GitHub issue:  
  👉 https://github.com/Insta360Develop/Desktop-CameraSDK-Cpp/issues/63  
  This involves downloading and installing the libUSBK drivers.

### 2. 🧰 Download the Camera and Media SDKs from drive

Place both SDK folders (CameraSDK and MediaSDK) under the directory (e.g., `codes/insta360/sdk/`).

---

### 3. ⚙️ Configure with CMake (Release Mode)

**Important**: Always compile in **Release** mode for SDK to function correctly.

See:  
👉 https://github.com/Insta360Develop/Desktop-CameraSDK-Cpp/issues/26

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="C:/path/to/opencv/build"
cmake --build . --config Release
```

Ensure that:
- Your system `PATH` contains the SDK `bin` directories

In case of any issues:
- Place the required `.dll` files (e.g., `CameraSDK.dll`, `MediaSDK.dll`) are placed in the same folder as your `.exe`


---

## ✅ Verifying Setup

Run your compiled program (e.g., `capture_and_stitch_realtime.exe`) from the `Release` folder:

```bash
cd build/Release
./capture_and_stitch_realtime.exe --debug
```

You should see logs like:
```
>>> Starting Insta360 SDK minimal test
>>> Found 1 camera(s)
>>> Succeed to open camera...
```

---

## 📝 Notes

- Use **Control with Android** mode on your Insta360 camera
- If no cameras are found, check USB mode and driver
- For stitching, initialize the `RealTimeStitcher` and feed in frames from the `StreamDelegate`

---

## 📁 Suggested Folder Layout

```
codes/
└── insta360/
    ├── build/
    ├── sdk/
    │   ├── CameraSDK/
    │   └── MediaSDK/
    └── capture_and_stitch_realtime.cpp
```