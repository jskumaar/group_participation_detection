# Group Participation Detection (C++ Implementation)

This project implements a 360-degree panoramic video gaze tracking system. It processes panoramic videos to track people and estimate their gaze direction to determine when individuals are looking at each other, facilitating group participation detection.

## Dependencies

To build and run this project, you need the following dependencies:

*   **CMake** (Version 3.10 or higher)
*   **C++ Compiler** (Supporting C++17)
*   **OpenCV**: Required for image processing.
    *   The project currently expects OpenCV to be located at `C:/opencv/build`. You may need to adjust `CMakeLists.txt` if your installation is elsewhere.
*   **ONNX Runtime**: Required for running the neural network models.
    *   The project expects the ONNX Runtime headers and libraries to be in an `onnx/` subdirectory within the project root (`src/c_impl/onnx`).
*   **Gnuplot**: Required for 3D visualization of the tracking and gaze vectors.
    *   Ensure `gnuplot` is in your system PATH.

### Models

You must have a `models/` directory in the project root (`src/c_impl/models`) containing the following ONNX models:

*   `head-localizer.onnx`
*   `head-pose-0.3-big-quantized.onnx`
*   `yolo11n.onnx`

## Folder Structure

```text
src/c_impl/
├── models/                 # Directory containing ONNX models
├── onnx/                   # ONNX Runtime library and headers
├── sort-cpp/               # SORT tracking algorithm implementation
├── 360_image_process.cpp   # 360-degree image processing logic
├── 360_image_process.h
├── CMakeLists.txt          # CMake build configuration
├── main.cpp                # Main application entry point
├── model_core.cpp          # Core model inference logic
├── model_core.h
├── SORT.cpp                # SORT tracker wrapper
├── SORT.h
├── tracker.cpp             # Object tracking implementation
├── tracker.h
└── ...
```

## Build Instructions

1.  Create a build directory:
    ```bash
    mkdir build
    cd build
    ```

2.  Generate the build files using CMake:
    ```bash
    cmake ..
    ```

3.  Build the project:
    ```bash
    cmake --build . --config Release
    ```

## Usage

After building, run the executable generated in the `build` directory. Ensure the `models` folder and the video file (configured in `main.cpp`) are accessible relative to the executable or update the paths in the code.
