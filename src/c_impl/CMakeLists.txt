cmake_minimum_required(VERSION 3.10)
project(tracker)

set(CMAKE_CXX_STANDARD 17)

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

# Sources
set(SOURCES
    model_core.cpp
    tracker.cpp
    SORT.cpp
    360_image_process.cpp
    #model_test.cpp
    main.cpp
    ${CMAKE_SOURCE_DIR}/sort-cpp/sort-c++/KalmanTracker.cpp
    ${CMAKE_SOURCE_DIR}/sort-cpp/sort-c++/Hungarian.cpp
)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories - add the onnxruntime subdirectory
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  onnxruntime
)