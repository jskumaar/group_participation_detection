# Group Participation Detection (C++ Implementation)

This project implements a 360-degree panoramic video gaze tracking system. It processes panoramic videos to track people and estimate their gaze direction to determine when individuals are looking at each other, facilitating group participation detection. The application provides a Qt-based GUI for video playback and gaze visualization (including VTK-based 3D visualization).

## Dependencies

Build and run require the following. Paths shown are those used in `CMakeLists.txt`; adjust them if your installations are elsewhere.

| Dependency | Version / Notes | CMake path (if set) |
|------------|-----------------|---------------------|
| **CMake** | 3.10 or higher | — |
| **C++ compiler** | C++17 (e.g. MSVC 2022 64-bit for Windows) | — |
| **OpenCV** | Required for image processing | `C:/opencv/build` |
| **ONNX Runtime** | Required for neural network inference | Headers: `onnx/include`, libs: `onnx/lib` (under project root) |
| **Qt** | Qt 6, Widgets component | `C:/Qt/6.10.1/msvc2022_64` |
| **VTK** | Required for 3D gaze visualization | `C:/VTK/install/lib/cmake/vtk-9.5` |
| **Gnuplot** | Optional; for standalone 3D visualization scripts | Ensure `gnuplot` is in your PATH if used |

### ONNX Runtime setup

Place ONNX Runtime headers and libraries under the project root:

- `src/c_impl/onnx/include/` — headers  
- `src/c_impl/onnx/lib/` — libraries (e.g. `onnxruntime.lib` for linking; you may need `onnxruntime.dll` next to the executable at run time)

Use the same ABI as your compiler (e.g. 64-bit MSVC build on Windows).

### Models

Create a `models/` directory in the project root (`src/c_impl/models/`) and add these ONNX models:

- `head-localizer.onnx`
- `head-pose-0.3-big-quantized.onnx`
- `yolo11n.onnx`

## Folder structure

```text
src/c_impl/
├── models/                 # ONNX models (see above)
├── onnx/                   # ONNX Runtime include/ and lib/
├── sort-cpp/               # SORT tracking implementation
├── 360_image_process.cpp  # 360° image processing
├── 360_image_process.h
├── CMakeLists.txt
├── main.cpp                # Application entry (Qt)
├── mainwindow.cpp / .h     # Main window UI
├── videowidget.cpp / .h    # Video display widget
├── gazevisualizer.cpp / .h # Gaze visualization (VTK)
├── model_core.cpp / .h     # Model inference
├── tracker.cpp / .h        # Object tracking
├── SORT.cpp / .h           # SORT tracker wrapper
└── ...
```

## Build instructions

1. **Configure paths (optional)**  
   If your OpenCV, Qt, or VTK are not at the default locations, edit `CMakeLists.txt`:
   - `OpenCV_DIR` (e.g. `C:/opencv/build`)
   - `CMAKE_PREFIX_PATH` for Qt (e.g. `C:/Qt/6.10.1/msvc2022_64`)
   - `VTK_DIR` (e.g. `C:/VTK/install/lib/cmake/vtk-9.5`)

2. **Create and enter a build directory**
   ```bash
   mkdir build
   cd build
   ```

3. **Configure with CMake**
   ```bash
   cmake ..
   ```
   On Windows with Visual Studio generator you may use:
   ```bash
   cmake -G "Visual Studio 17 2022" -A x64 ..
   ```

4. **Build**
   ```bash
   cmake --build . --config Release
   ```
   Or open the generated solution (e.g. `build/tracker.sln`) in Visual Studio and build from there.

## How to run

1. **Run from the build directory (recommended)**  
   From the `build` folder:
   ```bash
   .\Release\tracker.exe
   ```
   Or:
   ```bash
   .\x64\Release\tracker.exe
   ```
   (Exact path depends on your CMake generator; check under `build/` for `tracker.exe`.)

2. **Required at run time**
   - **Models**: The executable must find the `models/` folder. Either run from a directory where `models/` is in the expected place (e.g. copy or symlink `src/c_impl/models` next to `tracker.exe` or set the working directory to `src/c_impl`), or ensure the code is configured to look for `models/` via an absolute path or environment variable.
   - **DLLs**: Qt and VTK (and optionally OpenCV/ONNX Runtime if built as DLLs) must be on the PATH or next to `tracker.exe`. You can use `windeployqt` for Qt:
     ```bash
     C:\Qt\6.10.1\msvc2022_64\bin\windeployqt.exe .\Release\tracker.exe
     ```
   - **Video path**: Open your panoramic video file from the application UI, or set the default path in the code if applicable.

3. **Running from Visual Studio**  
   Set the project’s working directory to `src/c_impl` (or the folder that contains `models/`) so that relative paths to `models/` resolve correctly.

## Usage

After building and running, use the GUI to open a 360° panoramic video. The application will run head detection, tracking, and gaze estimation and show results in the video and gaze visualizer. Ensure the `models/` directory and any required DLLs are available as described above.
