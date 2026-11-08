# Group Participation Detection (C++)

Qt desktop app for 360° panoramic video: head detection, SORT tracking, gaze estimation (ONNX), and VTK-based gaze visualization. Optional gRPC streaming publishes pipeline updates (`proto/rapport.proto`).

## Prerequisites (Windows)

| Dependency | Notes | Default path in `CMakeLists.txt` |
|------------|-------|----------------------------------|
| **Visual Studio 2022** | Desktop development with C++ workload | — |
| **CMake** | 3.10+ | — |
| **OpenCV** | 4.x, x64 MSVC build | `C:/opencv/build` |
| **Qt 6** | Widgets module, MSVC 2022 64-bit | `C:/Qt/6.10.1/msvc2022_64` |
| **VTK** | 9.x, built for same MSVC/Qt | `C:/VTK/install/lib/cmake/vtk-9.5` |
| **ONNX Runtime** | 1.22.1, MSVC x64; not in git, see below | `onnx/include`, `onnx/lib` |
| **vcpkg** | `grpc` and `protobuf` for C++ | Auto-used if `VCPKG_ROOT` is set (e.g. `C:/vcpkg`) |

Edit `CMakeLists.txt` if OpenCV, Qt, or VTK live elsewhere (`OpenCV_DIR`, `CMAKE_PREFIX_PATH`, `VTK_DIR`).

### vcpkg (gRPC / Protobuf)

If CMake cannot find `protobuf` or `gRPC`:

```powershell
git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg install grpc:x64-windows protobuf:x64-windows
```

Ensure `VCPKG_ROOT` points at your vcpkg root (CMake integrates via `%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake` when the toolchain is active).

### ONNX Runtime

Not in git: `onnx/` is listed in `.gitignore`. Download the Windows x64 archive for version 1.22.1 from the [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases) and unpack it so that these exist:

- `onnx/include/` — headers  
- `onnx/lib/onnxruntime.lib` — link library  

At run time, copy `onnxruntime.dll` from `onnx/lib/` next to `tracker.exe`. Adding it to `PATH` is not enough: Windows ships its own `onnxruntime.dll` in System32 and finds that one first, which crashes the app at startup with an access violation (exit code 0xC0000005).

### ONNX models

Not in git: `models/` is listed in `.gitignore`. Place these in `c_impl/models/`. The app finds them relative to the working directory, so see [Run](#run):

| File | Purpose |
|------|---------|
| `head-localizer.onnx` | Head localization |
| `head-pose-0.3-big-quantized.onnx` | Head pose |
| `L2CSNet_gaze360.onnx` | Gaze (L2CS-Net) |
| `yolo11n.onnx` | Person detection |

## Project layout

```text
c_impl/
├── app/              # Qt UI (main, mainwindow, videowidget, gazevisualizer)
├── vision/           # Models, tracking, 360° processing
├── pipelines/        # Vision pipeline orchestration
├── media/            # Video input
├── io/               # CSV export, gRPC publisher/service
├── proto/            # rapport.proto
├── third_party/      # SORT (Kalman + Hungarian)
├── models/           # ONNX weights (not in git — see table above)
├── onnx/             # ONNX Runtime SDK, include + lib (not in git)
├── CMakeLists.txt
└── build/            # Out-of-source build (created by you)
```

## Build

All commands assume PowerShell and that you start in `src/c_impl`.

### Option A — Command line (CMake + MSVC)

```powershell
cd C:\Python\Research\group_participation_detection\src\c_impl
mkdir build -Force
cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

Output: `build\Release\tracker.exe`

### Option B — Visual Studio (CMake integration)

1. Open the folder `src/c_impl` in Visual Studio (**File → Open → CMake…**), or open `CMakeLists.txt`.
2. Select configuration **x64-Release** (see `CMakeSettings.json`; uses Ninja + RelWithDebInfo).
3. **Build → Build All**.

Output is under `out/build/x64-Release/` (or the path shown in the CMake build log).

### Clean reconfigure

```powershell
Remove-Item -Recurse -Force build
mkdir build; cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

## Run

1. **Working directory / models**  
   The app loads ONNX files from `../../models/`, relative to the **working directory** and not to the executable. Start it from `build\Release` so that this resolves to `c_impl/models/`. From anywhere else it exits at startup. In Visual Studio, set the debugging working directory to `build\Release`, or to another folder two levels below `c_impl`.

   The app also overwrites `gaze_analysis.csv` in the working directory at every launch.

2. **Deploy Qt DLLs** (if the app fails to start with missing Qt plugins):

   ```powershell
   C:\Qt\6.10.1\msvc2022_64\bin\windeployqt.exe .\build\Release\tracker.exe
   ```

3. **Other DLLs**  
   Ensure OpenCV and VTK DLLs are on `PATH` or beside `tracker.exe`. `onnxruntime.dll` must be beside `tracker.exe`, as described under [ONNX Runtime](#onnx-runtime).

4. **Launch**

   ```powershell
   cd build\Release
   .\tracker.exe
   ```

   Use the GUI to open a 360° video file.

## Troubleshooting

| Symptom | What to check |
|---------|----------------|
| `Cannot find source file ... KalmanTracker.cpp` | SORT sources must be `third_party/KalmanTracker.cpp` and `Hungarian.cpp` (included in repo). |
| `find_package(protobuf)` / `gRPC` fails | Install via vcpkg; set `VCPKG_ROOT`. |
| Missing ONNX models at runtime | Start the app from `build\Release`, and check that `c_impl/models/` has the files. |
| `onnxruntime.dll` not found, or an access violation at startup | Copy `onnx/lib/onnxruntime.dll` next to `tracker.exe`. |
| "gRPC stream unavailable" at startup | Something else is using the port, often another `tracker.exe`. Close it, or set `RAPPORT_GRPC_TARGET` to another address. |
| Qt platform plugin errors | Run `windeployqt` on `tracker.exe`. |

## Usage

Open a panoramic video in the UI. The pipeline runs detection, tracking, and gaze estimation; results appear on the video view and in the VTK gaze visualizer. CSV and gRPC outputs are handled under `io/` when enabled in the application flow.
