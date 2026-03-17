cmake_minimum_required(VERSION 3.10)
project(tracker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

if(MSVC)
  # Avoid MSVC "cannot open program database" (C1041) under parallel builds.
  add_compile_options(/FS)
endif()

# ONNX Runtime - use the correct include path
include_directories(
  ${CMAKE_SOURCE_DIR}/onnx/include
)


link_directories(
  ${CMAKE_SOURCE_DIR}/onnx/lib
)

# OpenCV
set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Qt
set(CMAKE_PREFIX_PATH "C:/Qt/6.10.1/msvc2022_64")
find_package(Qt6 COMPONENTS Widgets REQUIRED)

# VTK
set(VTK_DIR "C:/VTK/install/lib/cmake/vtk-9.5")
find_package(VTK REQUIRED)

# Sources
set(SOURCES
    vision/model_core.cpp
    vision/opnet_tracker.cpp
    vision/sort_tracker.cpp
    vision/360_image_process.cpp
    #model_test.cpp
    app/main.cpp
    app/app_controller.cpp
    app/mainwindow.cpp
    app/videowidget.cpp
    app/gazevisualizer.cpp
    media/video_source.cpp
    io/csv_gaze_writer.cpp
    pipelines/vision_pipeline.cpp
    ${CMAKE_SOURCE_DIR}/third_party/sort-cpp/sort-c++/KalmanTracker.cpp
    ${CMAKE_SOURCE_DIR}/third_party/sort-cpp/sort-c++/Hungarian.cpp
)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Include directories - add the onnxruntime subdirectory
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  onnxruntime
  Qt6::Widgets
  ${VTK_LIBRARIES}
)