# Group participation detection — `src`

This folder contains the main C++ application and related code.

## Build the desktop app

The Qt/ONNX tracker GUI lives in **`c_impl/`**. See **[c_impl/README.md](c_impl/README.md)** for dependencies, CMake steps, and how to run `tracker.exe`.

Quick start (Windows, from `c_impl`):

```powershell
mkdir build -Force; cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

Executable: `c_impl/build/Release/tracker.exe` (copy `models/` beside it before running).
