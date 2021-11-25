cmake_minimum_required(VERSION 3.1.0)
project(lins)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_BUILD_TYPE Release)

# set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/lib/x86_64-linux-gnu/cmake")
# set(OpenCV_DIR "/usr/local/share/OpenCV")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/usr/share/cmake/geographiclib/")

find_package(catkin REQUIRED COMPONENTS
    cloud_msgs
    cv_bridge
    geometry_msgs
    image_transport
    nav_msgs
    pcl_conversions
    pcl_ros
    roscpp
    rospy
    sensor_msgs
    std_msgs
    tf
    sleipnir_msgs
    # livox_ros_driver
)

find_package(GTSAM REQUIRED QUIET)
find_package(PCL REQUIRED QUIET)
find_package(OpenCV 3 REQUIRED QUIET)
find_package(Eigen3 3.3 REQUIRED)
find_package (GeographicLib REQUIRED)

# message(WARNING "   OpenCV library status:")
# message(WARNING "    config: ${OpenCV_DIR}")
# message(WARNING "    version: ${OpenCV_VERSION}")
# message(WARNING "    libraries: ${OpenCV_LIBS}")
# message(WARNING "    include path: ${OpenCV_INCLUDE_DIRS}")

catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS cloud_msgs
    DEPENDS PCL
)


include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

link_directories(
    include
    ${OpenCV_LIBRARY_DIRS}
    ${PCL_LIBRARY_DIRS}
    ${GTSAM_LIBRARY_DIRS}
)


list(APPEND SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/lib/parameters.cpp
)

list(APPEND LINS_FILES
    ${PROJECT_SOURCE_DIR}/src/lins_fusion_node.cpp
    ${PROJECT_SOURCE_DIR}/src/lib/Estimator.cpp
)

list(APPEND LINK_LIBS 
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARY_DIRS}
    ${GeographicLib_LIBRARIES}
)

add_executable(lins_fusion_node ${LINS_FILES} ${SOURCE_FILES})
target_link_libraries(lins_fusion_node ${LINK_LIBS})

add_executable(image_projection_node src/image_projection_node.cpp ${SOURCE_FILES})
# add_dependencies(image_projection_node ${catkin_EXPORTED_TARGETS} cloud_msgs_gencpp)
target_link_libraries(image_projection_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(lidar_mapping_node src/lidar_mapping_node.cpp ${SOURCE_FILES} src/lib/Scancontext.cpp src/LidarIris.cpp src/fftm.cpp)
target_link_libraries(lidar_mapping_node ${LINK_LIBS} gtsam Eigen3::Eigen)

add_executable(transform_fusion_node src/transform_fusion_node.cpp ${SOURCE_FILES})
target_link_libraries(transform_fusion_node ${LINK_LIBS})

add_executable(mapLocalization src/mapLocalization.cpp ${SOURCE_FILES})
target_link_libraries(mapLocalization ${LINK_LIBS} gtsam Eigen3::Eigen)

add_executable(ImuFactorDemo
    src/ImuFactorDemo.cpp
)
target_link_libraries(ImuFactorDemo 
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES} 
    ${OpenCV_LIBRARIES} 
    gtsam
    Eigen3::Eigen
)

add_executable(staticTFpub_node src/staticTFpub_node.cpp src/lib/XmlDomDocument.cpp ${SOURCE_FILES})
target_link_libraries(staticTFpub_node ${catkin_LIBRARIES} -lxerces-c)