cmake_minimum_required(VERSION 2.8.3)
project(kinect2_viewer)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBT_USE_DOUBLE_PRECISION -Wall")
# Unused warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized -Winit-self -Wunused-function -Wunused-label -Wunused-variable -Wunused-but-set-variable -Wunused-but-set-parameter")
# Additional warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Warray-bounds -Wtype-limits -Wreturn-type -Wsequence-point -Wparentheses -Wmissing-braces -Wchar-subscripts -Wswitch -Wwrite-strings -Wenum-compare -Wempty-body -Wlogical-op")

#set(CMAKE_PREFIX_PATH "/usr/lib/)
#set(CMAKE_PREFIX_PATH /usr/lib/x86_64-linux-gnu)
#set(CMAKE_PREFIX_PATH /usr/local/lib)

OPTION(ENABLE_CXX11 "Enable C++11 support" ON)

# Check for c++11 support
INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
IF(COMPILER_SUPPORTS_CXX11)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ELSEIF(COMPILER_SUPPORTS_CXX0X)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
ELSE()
  MESSAGE(ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
ENDIF()

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  #Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

set(MY_DIR ${kinect2_viewer_SOURCE_DIR})

find_package(catkin REQUIRED COMPONENTS roscpp rostime std_msgs sensor_msgs message_filters cv_bridge image_transport compressed_image_transport compressed_depth_image_transport kinect2_bridge)

## System dependencies are found with CMake's conventions
find_package(PkgConfig)
find_package(OpenCV REQUIRED)
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
find_package(OpenMP)
find_package(PCL REQUIRED)

if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

################################################
## Declare ROS messages, services and actions ##
#message(\nOpenMP CXX Flags: ${OpenMP_CXX_FLAGS}")
################################################

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES kinect2_viewer
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

# Display additional files in qtcreator
execute_process(COMMAND find ${PROJECT_SOURCE_DIR} -type f OUTPUT_VARIABLE FILES_STRING)
string(REPLACE "\n" ";" FILES_LIST ${FILES_STRING})
add_custom_target(additional_files_${PROJECT_NAME}
  SOURCES
  ${FILES_LIST}
)

include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${kinect2_bridge_INCLUDE_DIRS}
  "${MY_DIR}/include"
)

add_definitions(${EIGEN_DEFINITIONS})

LINK_DIRECTORIES(${OpenCV_LIBS})

add_executable(kinect2_viewer src/viewer.cpp)



TARGET_LINK_LIBRARIES(kinect2_viewer ${OpenCV_LIBRARIES}
#                                     ${OpenCV_LIBS}
                                     ${catkin_LIBRARIES}
                                     ${PCL_LIBRARIES}
                                     ${kinect2_bridge_LIBRARIES}
#                                     "/usr/lib/x86_64-linux-gnu/libOpenCL.so"      
                      )

#message(STATUS             "\n\nSavitzky include path: \n\n")
#message(STATUS               "${MY_DIR}/include")
#message(STATUS              "\n\nCATKIN LIBRARIES: \n\n  ${catkin_LIBRARIES}
#                             \n\n OpenCV LIBRARIES: \n\n ${OpenCV_LIBRARIES}
#                             \n\n PCL_LIBRARIES: \n\n    ${PCL_LIBRARIES}
#                             \n\n KINECT2_BRIDGE LIBRARIES: \n\n ${kinect2_bridge_LIBRARIES}")

message(STATUS "\n\nEigen3 include path: ${EIGEN3_INCLUDE_DIRS}\n")
message(STATUS "\n\nEigen3 Libraries: ${EIGEN3_LIBRARIES}\n")

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
install(TARGETS kinect2_viewer
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_kinect2_bridge.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
