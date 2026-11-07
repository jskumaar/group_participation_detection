cmake_minimum_required(VERSION 3.10)
project(port_check)
set(CMAKE_CXX_STANDARD 17)
if(NOT C_IMPL)
  message(FATAL_ERROR "pass -DC_IMPL=<path to src/c_impl>")
endif()
set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)
add_executable(port_check port_check.cpp
  ${C_IMPL}/vision/gaze_bayes_mapper.cpp
  ${C_IMPL}/vision/360_image_process.cpp)
target_include_directories(port_check PRIVATE ${C_IMPL} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(port_check ${OpenCV_LIBS})
