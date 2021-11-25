// This file is part of LINS.
//
// Copyright (C) 2020 Chao Qin <cscharlesqin@gmail.com>,
// Robotics and Multiperception Lab (RAM-LAB <https://ram-lab.com>),
// The Hong Kong University of Science and Technology
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.

#ifndef INCLUDE_PARAMETERS_H_
#define INCLUDE_PARAMETERS_H_

#include <math.h>
#include <nav_msgs/Odometry.h>
#include <pcl/common/common.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tic_toc.h>

#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <mutex>
#include <flann/flann.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "cloud_msgs/cloud_info.h"

#include <fstream>

#include <cstdlib>
#include <ctime>

#include <GeographicLib/LambertConformalConic.hpp> 
#include <GeographicLib/Geodesic.hpp>
#include <GeographicLib/UTMUPS.hpp>
#include <GeographicLib/MGRS.hpp>


#define PI (3.14159265)

using namespace std;

//the following are UBUNTU/LINUX ONLY terminal color codes.
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */


struct smoothness_t {
  float value;
  size_t ind;
};

struct by_value {
  bool operator()(smoothness_t const &left, smoothness_t const &right) {
    return left.value < right.value;
  }
};

/*
    * A point cloud type that has "ring" channel
    */
struct PointXYZIR
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIR,  
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (uint16_t, ring, ring)
)

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time)
)

typedef PointXYZIRPYT  PointTypePose;
// --------------------------------------------------------------------------------------------------------

typedef pcl::PointXYZI PointType;

typedef Eigen::Vector3d V3D;
typedef Eigen::Matrix3d M3D;
typedef Eigen::VectorXd VXD;
typedef Eigen::MatrixXd MXD;
typedef Eigen::Quaterniond Q4D;

namespace parameter {

/*!@EARTH COEFFICIENTS */
const double G0 = 9.81;                  // gravity
const double deg = M_PI / 180.0;         // degree
const double rad = 180.0 / M_PI;         // radian
const double dph = deg / 3600.0;         // degree per hour
const double dpsh = deg / sqrt(3600.0);  // degree per square-root hour
const double mg = G0 / 1000.0;           // mili-gravity force
const double ug = mg / 1000.0;           // micro-gravity force
const double mgpsHz = mg / sqrt(1.0);    // mili-gravity force per second
const double ugpsHz = ug / sqrt(1.0);    // micro-gravity force per second
const double Re = 6378137.0;             ///< WGS84 Equatorial radius in meters
const double Rp = 6356752.31425;
const double Ef = 1.0 / 298.257223563;
const double Wie = 7.2921151467e-5;
const double Ee = 0.0818191908425;
const double EeEe = Ee * Ee;

// IMU 内参
const float accel_noise_sigma = 3.9939570888238808e-03;
const float gyro_noise_sigma = 1.5636343949698187e-03;
const float accel_bias_rw_sigma = 6.4356659353532566e-05;
const float gyro_bias_rw_sigma = 3.5640318696367613e-05;
const float imuGravity = 9.80511;

/*!@SLAM COEFFICIENTS */
const bool loopClosureEnableFlag = true;
const double mappingProcessInterval = 0.3;

const float ang_res_x = 0.2;
const float ang_res_y = 2.0;
const float ang_bottom = 15.0 + 0.1;
const int groundScanInd = 5;

// HDL-32E
// const int N_SCAN = 32;
// const int Horizon_SCAN = 1800;
// const float ang_res_x = 0.2;
// const float ang_res_y = 1.33;
// const float ang_bottom = 30.67;
// const int groundScanInd = 7;

// const float ang_res_x = 0.2;
// const float ang_res_y = 0.427;
// const float ang_bottom = 24.9;
// const int groundScanInd = 50;

const int systemDelay = 0;
const float sensorMountAngle = 0.0;
const float segmentTheta = 1.0472;
const int segmentValidPointNum = 5;
const int segmentValidLineNum = 3;
const float segmentAlphaX = ang_res_x / 180.0 * M_PI;
const float segmentAlphaY = ang_res_y / 180.0 * M_PI;
const int edgeFeatureNum = 2;
const int surfFeatureNum = 4;
const int sectionsTotal = 6;
const float surroundingKeyframeSearchRadius = 50.0;
const int surroundingKeyframeSearchNum = 50;
const float historyKeyframeSearchRadius = 10.0;
const int historyKeyframeSearchNum = 25;
const float historyKeyframeFitnessScore = 0.08;
const float globalMapVisualizationSearchRadius = 1500.0;

// !@ENABLE_CALIBRATION
extern int CALIBARTE_IMU;
extern int SHOW_CONFIGURATION;
extern int AVERAGE_NUMS;

// !@INITIAL_PARAMETERS
extern double IMU_LIDAR_EXTRINSIC_ANGLE;
extern double IMU_MISALIGN_ANGLE;

// !@LIDAR_PARAMETERS
extern int LINE_NUM;
extern int SCAN_NUM;
extern double SCAN_PERIOD;
extern double EDGE_THRESHOLD;
extern double SURF_THRESHOLD;
extern double NEAREST_FEATURE_SEARCH_SQ_DIST;

// !@TESTING
extern int VERBOSE;
extern int ICP_FREQ;
extern int MAX_LIDAR_NUMS;
extern int NUM_ITER;
extern double LIDAR_SCALE;
extern double LIDAR_STD;

// !@SUB_TOPIC_NAME
extern std::string IMU_TOPIC;
extern std::string LIDAR_TOPIC;

// !@PUB_TOPIC_NAME
extern std::string LIDAR_ODOMETRY_TOPIC;
extern std::string LIDAR_MAPPING_TOPIC;

// !@KALMAN_FILTER
extern double ACC_N;
extern double ACC_W;
extern double GYR_N;
extern double GYR_W;
extern V3D INIT_POS_STD;
extern V3D INIT_VEL_STD;
extern V3D INIT_ATT_STD;
extern V3D INIT_ACC_STD;
extern V3D INIT_GYR_STD;

// !@INITIAL IMU BIASES
extern V3D INIT_BA;
extern V3D INIT_BW;

// !@EXTRINSIC_PARAMETERS
extern V3D INIT_TBL;
extern Q4D INIT_RBL;

extern double OriLon;
extern double OriLat;
extern double OriAlt;
extern double OriYaw;
extern double OriPitch;
extern double OriRoll;
extern double compensate_init_yaw;
extern double compensate_init_pitch;
extern double compensate_init_roll;
extern double mappingCarYawPara;
extern double InitPose_x;
extern double InitPose_y;
extern double InitPose_yaw;

void readParameters(ros::NodeHandle& n);
void readInitPose(ros::NodeHandle& n);

void readV3D(cv::FileStorage* file, const std::string& name, V3D& vec_eigen);

void readQ4D(cv::FileStorage* file, const std::string& name, Q4D& quat_eigen);

enum StateOrder {
  O_R = 0,
  O_P = 3,
};

}  // namespace parameter

#endif  // INCLUDE_PARAMETERS_H_
