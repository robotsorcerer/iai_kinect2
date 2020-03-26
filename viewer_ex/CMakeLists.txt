# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.9.0)
set(AZURE_DIR $ENV{HOME}/Documents/azure-kinect)

set(INCLUDE_DIR ${AZURE_DIR}/include)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenGL REQUIRED)
include_directories(
    include
		${AZURE_DIR}/include
		${AZURE_DIR}/build/src/sdk/include/
		${AZURE_DIR}/extern/imgui/src/examples/libs/gl3w/
		${AZURE_DIR}/extern/glfw/src/include/
		${AZURE_DIR}/extern/imgui/src/
		${AZURE_DIR}/extern/imgui/src/examples/
    ${OPENGL_INCLUDE_DIRS}
)



add_executable(viewer_gl src/main.cpp
												     src/texture.cpp
												     src/viewerwindow.cpp)

link_directories(${AZURE_DIR}/build/bin ${AZURE_DIR}/build/extern/imgui/)

target_link_libraries(viewer_gl PRIVATE
    k4a #::k4a
    glfw #::glfw
		${OPENGL_LIBRARIES}
		${AZURE_DIR}/build/extern/imgui/libimgui.a
    ${OPENGL_LIBRARIES})
