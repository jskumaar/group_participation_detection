cmake_minimum_required(VERSION 3.15)
project(tracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime - use the correct include path
set(ONNXRUNTIME_DIR "/opt/homebrew/opt/onnxruntime")

# OpenCV
find_package(OpenCV REQUIRED)


set(SORT_SOURCES 
${CMAKE_SOURCE_DIR}/sort-cpp/sort-c++/KalmanTracker.cpp
${CMAKE_SOURCE_DIR}/sort-cpp/sort-c++/Hungarian.cpp
)
# Sources
set(SOURCES
    model_core.cpp
    tracker.cpp
    360_image_process.cpp
    main.cpp
    SORT.cpp
)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES} ${SORT_SOURCES})

# Include directories - add the onnxruntime subdirectory
target_include_directories(${PROJECT_NAME} PRIVATE
    ${ONNXRUNTIME_DIR}/include/onnxruntime
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/c_impl 
    ${CMAKE_SOURCE_DIR}/c_impl/sort-cpp/sort-c++
)


target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib
)