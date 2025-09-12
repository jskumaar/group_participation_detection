cmake_minimum_required(VERSION 3.10)
project(tracker)

set(CMAKE_CXX_STANDARD 17)

# SDK Includes
include_directories(
  ${CMAKE_SOURCE_DIR}/onnx/include
)

# SDK Libs
link_directories(
  ${CMAKE_SOURCE_DIR}/onnx/lib
)

# OpenCV
set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SORT_SOURCES 
${CMAKE_SOURCE_DIR}/sort-cpp/sort-c++/KalmanTracker.cpp
${CMAKE_SOURCE_DIR}/sort-cpp/sort-c++/Hungarian.cpp
)

set(SOURCES
  model_core.cpp
  tracker.cpp
  360_image_process.cpp
  SORT.cpp
  main.cpp
#   model_test.cpp
)



# Executable
add_executable(${PROJECT_NAME} ${SOURCES} ${SORT_SOURCES})
# add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  onnxruntime
  ${CMAKE_SOURCE_DIR}/sort-cpp/sort-c++
)