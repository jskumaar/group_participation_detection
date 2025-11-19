cmake_minimum_required(VERSION 3.10)
project(capture_and_stitch_realtime)

set(CMAKE_CXX_STANDARD 17)

# SDK Includes
include_directories(
  ${CMAKE_SOURCE_DIR}/sdk/CameraSDK/include
  ${CMAKE_SOURCE_DIR}/sdk/MediaSDK/include
)

# SDK Libs
link_directories(
  ${CMAKE_SOURCE_DIR}/sdk/CameraSDK/lib
  ${CMAKE_SOURCE_DIR}/sdk/MediaSDK/lib
)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Executable
add_executable(${PROJECT_NAME} capture_and_stitch_realtime.cpp)

# Link libraries
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  CameraSDK
  MediaSDK
)
