cmake_minimum_required(VERSION 3.10)

# vcpkg toolchain for protobuf/gRPC — must be set before project()
if(NOT CMAKE_TOOLCHAIN_FILE)
  if(DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE FILEPATH "vcpkg toolchain file")
  elseif(EXISTS "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE FILEPATH "vcpkg toolchain file")
  endif()
endif()

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

# Help CMake locate vcpkg-installed protobuf/gRPC (and their deps)
set(_vcpkg_installed "")
if(DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}/installed/x64-windows")
  set(_vcpkg_installed "$ENV{VCPKG_ROOT}/installed/x64-windows")
elseif(EXISTS "C:/vcpkg/installed/x64-windows")
  set(_vcpkg_installed "C:/vcpkg/installed/x64-windows")
endif()
if(_vcpkg_installed)
  list(APPEND CMAKE_PREFIX_PATH "${_vcpkg_installed}")
  if(EXISTS "${_vcpkg_installed}/share/protobuf/protobuf-config.cmake")
    set(protobuf_DIR "${_vcpkg_installed}/share/protobuf" CACHE PATH "vcpkg protobuf")
  endif()
  if(EXISTS "${_vcpkg_installed}/share/grpc/gRPCConfig.cmake")
    set(gRPC_DIR "${_vcpkg_installed}/share/grpc" CACHE PATH "vcpkg gRPC")
  endif()
endif()

find_package(protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

set(RAPPORT_PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/proto/rapport.proto)
set(RAPPORT_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${RAPPORT_GENERATED_DIR})

set(RAPPORT_PROTO_CC ${RAPPORT_GENERATED_DIR}/rapport.pb.cc)
set(RAPPORT_PROTO_H ${RAPPORT_GENERATED_DIR}/rapport.pb.h)
set(RAPPORT_GRPC_CC ${RAPPORT_GENERATED_DIR}/rapport.grpc.pb.cc)
set(RAPPORT_GRPC_H ${RAPPORT_GENERATED_DIR}/rapport.grpc.pb.h)

add_custom_command(
  OUTPUT ${RAPPORT_PROTO_CC} ${RAPPORT_PROTO_H} ${RAPPORT_GRPC_CC} ${RAPPORT_GRPC_H}
  COMMAND $<TARGET_FILE:protobuf::protoc>
  ARGS
    --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
    --cpp_out=${RAPPORT_GENERATED_DIR}
    --grpc_out=${RAPPORT_GENERATED_DIR}
    --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
    ${RAPPORT_PROTO_FILE}
  DEPENDS ${RAPPORT_PROTO_FILE}
  COMMENT "Generating C++ protobuf/gRPC sources for rapport.proto"
  VERBATIM
)

set(RAPPORT_GENERATED_SRCS
  ${RAPPORT_PROTO_CC}
  ${RAPPORT_GRPC_CC}
)

# Sources
set(SOURCES
    vision/model_core.cpp
    vision/opnet_tracker.cpp
    vision/sort_tracker.cpp
    vision/360_image_process.cpp
    vision/gaze_bayes_mapper.cpp
    app/main.cpp
    app/app_controller.cpp
    app/mainwindow.cpp
    app/videowidget.cpp
    app/gazevisualizer.cpp
    media/video_source.cpp
    io/csv_gaze_writer.cpp
    io/grpc_pipeline_stream_service.cpp
    io/grpc_pipeline_publisher.cpp
    pipelines/vision_pipeline.cpp
    ${CMAKE_SOURCE_DIR}/third_party/KalmanTracker.cpp
    ${CMAKE_SOURCE_DIR}/third_party/Hungarian.cpp
    ${RAPPORT_GENERATED_SRCS}
)

# Executable (GUI subsystem on Windows: no extra console window)
add_executable(${PROJECT_NAME} WIN32 ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(${PROJECT_NAME} PRIVATE
  ${RAPPORT_GENERATED_DIR}
)

# Include directories - add the onnxruntime subdirectory
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  onnxruntime
  Qt6::Widgets
  ${VTK_LIBRARIES}
)

target_link_libraries(${PROJECT_NAME}
  protobuf::libprotobuf
  gRPC::grpc++
  gRPC::grpc
)