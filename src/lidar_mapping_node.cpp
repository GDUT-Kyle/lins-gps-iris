// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in CornerMap and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of CornerMap code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar
//   Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems
//      (IROS). October 2018.

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <math_utils.h>
#include <parameters.h>

#include <sensor_msgs/NavSatStatus.h>
#include <sensor_msgs/NavSatFix.h>
#include <gps_common/conversions.h>
#include <nav_msgs/Odometry.h>
#include <queue> // For queue container
#include <mutex>

#include <eigen3/Eigen/Dense>

#include <pcl/io/pcd_io.h>

#include "sc_lego_loam/Scancontext.h"
// #include "utility.h"

#include <sleipnir_msgs/sensorgps.h>

// #include "livox_ros_driver/CustomMsg.h"

using namespace gtsam;
using namespace parameter;
using namespace std;
using namespace gps_common;
using namespace GeographicLib;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

const int imuQueLength_ = 200;

class MappingHandler {
 private:
  NonlinearFactorGraph gtSAMgraph;
  Values initialEstimate;
  Values optimizedEstimate;
  ISAM2* isam;
  Values isamCurrentEstimate;

  noiseModel::Diagonal::shared_ptr priorNoise;
  noiseModel::Diagonal::shared_ptr odometryNoise;
  noiseModel::Diagonal::shared_ptr constraintNoise;
  noiseModel::Diagonal::shared_ptr LidarNoise;

  ros::NodeHandle nh;
  ros::NodeHandle pnh;

  ros::Publisher pubLaserCloudSurround;
  ros::Publisher pubOdomAftMapped;
  ros::Publisher pubKeyPoses;
  ros::Publisher pubOdomXYZAftMapped;

  ros::Publisher pubHistoryKeyFrames;
  ros::Publisher pubIcpKeyFrames;
  ros::Publisher pubRecentKeyFrames;

  ros::Publisher pubLoopConstraintEdge;

  ros::Subscriber subLaserCloudCornerLast;
  ros::Subscriber subLaserCloudSurfLast;
  ros::Subscriber subOutlierCloudLast;
  ros::Subscriber subLaserOdometry;
  ros::Subscriber subImu;

  ros::Subscriber sleipnir_gps_sub;

  ros::Subscriber sub_livox_msg1;

  tf::TransformBroadcaster tfBroadcaster;
  nav_msgs::Odometry odomAftMapped;
  tf::StampedTransform aftMappedTrans;

  tf::TransformBroadcaster tfXYZBroadcaster;
  nav_msgs::Odometry odomXYZAftMapped;
  tf::StampedTransform aftMappedXYZTrans;

  vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
  vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
  vector<pcl::PointCloud<PointType>::Ptr> outlierCloudKeyFrames;
  vector<pcl::PointCloud<PointType>::Ptr> originCloudKeyFrames;

  deque<pcl::PointCloud<PointType>::Ptr> recentCornerCloudKeyFrames;
  deque<pcl::PointCloud<PointType>::Ptr> recentSurfCloudKeyFrames;
  deque<pcl::PointCloud<PointType>::Ptr> recentOutlierCloudKeyFrames;
  int latestFrameID;

  vector<int> surroundingExistingKeyPosesID;
  deque<pcl::PointCloud<PointType>::Ptr> surroundingCornerCloudKeyFrames;
  deque<pcl::PointCloud<PointType>::Ptr> surroundingSurfCloudKeyFrames;
  deque<pcl::PointCloud<PointType>::Ptr> surroundingOutlierCloudKeyFrames;

  PointType previousRobotPosPoint;
  PointType currentRobotPosPoint;

  pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
  pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;

  pcl::PointCloud<PointType>::Ptr surroundingKeyPoses;
  pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS;

  pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
  pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;

  pcl::PointCloud<PointType>::Ptr laserCloudOutlierLast;
  pcl::PointCloud<PointType>::Ptr laserCloudOutlierLastDS;

  pcl::PointCloud<PointType>::Ptr laserCloudSurfTotalLast;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfTotalLastDS;

  pcl::PointCloud<PointType>::Ptr laserCloudOri;
  pcl::PointCloud<PointType>::Ptr coeffSel;

  pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
  pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

  pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

  pcl::PointCloud<PointType>::Ptr nearHistoryCornerKeyFrameCloud;
  pcl::PointCloud<PointType>::Ptr nearHistoryCornerKeyFrameCloudDS;
  pcl::PointCloud<PointType>::Ptr nearHistorySurfKeyFrameCloud;
  pcl::PointCloud<PointType>::Ptr nearHistorySurfKeyFrameCloudDS;

  pcl::PointCloud<PointType>::Ptr latestCornerKeyFrameCloud;
  pcl::PointCloud<PointType>::Ptr latestSurfKeyFrameCloud;
  pcl::PointCloud<PointType>::Ptr latestSurfKeyFrameCloudDS;

  pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap;
  pcl::PointCloud<PointType>::Ptr globalMapKeyPoses;
  pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS;
  pcl::PointCloud<PointType>::Ptr globalMapKeyFrames;
  pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr laserCloudFullResColor;

  pcl::PointCloud<PointType>::Ptr globalCornerMapKeyFrames;
  pcl::PointCloud<PointType>::Ptr globalSurfMapKeyFrames;

  std::vector<int> pointSearchInd;
  std::vector<float> pointSearchSqDis;

  pcl::VoxelGrid<PointType> downSizeFilterCorner;
  pcl::VoxelGrid<PointType> downSizeFilterSurf;
  pcl::VoxelGrid<PointType> downSizeFilterOutlier;
  pcl::VoxelGrid<PointType> downSizeFilterHistoryKeyFrames;
  pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;
  pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;
  pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;

  double timeLaserCloudCornerLast;
  double timeLaserCloudSurfLast;
  double timeLaserOdometry;
  double timeLaserCloudOutlierLast;
  double timeLastGloalMapPublish;
  double timeLaserCloudRawLast;

  bool newLaserCloudCornerLast;
  bool newLaserCloudSurfLast;
  bool newLaserOdometry;
  bool newLaserCloudOutlierLast;
  bool newLaserCloudRawLast;

  // scan context
  pcl::PointCloud<PointType>::Ptr RSlatestSurfKeyFrameCloud; // giseop, RS: radius search 
  pcl::PointCloud<PointType>::Ptr RSnearHistorySurfKeyFrameCloud;
  pcl::PointCloud<PointType>::Ptr RSnearHistorySurfKeyFrameCloudDS;

  pcl::PointCloud<PointType>::Ptr SClatestSurfKeyFrameCloud; // giseop, SC: scan context
  pcl::PointCloud<PointType>::Ptr SCnearHistorySurfKeyFrameCloud;
  pcl::PointCloud<PointType>::Ptr SCnearHistorySurfKeyFrameCloudDS;
  pcl::PointCloud<PointType>::Ptr laserCloudRaw; 
  pcl::PointCloud<PointType>::Ptr laserCloudRawDS; 
  pcl::VoxelGrid<PointType> downSizeFilterScancontext;
  ros::Subscriber subLaserCloudRaw;

  pcl::PointCloud<PointType>::Ptr OriginateGPStoMGRSforTest; 
  // pcl::PointCloud<PointType>::Ptr FusiontoUTMforTest; 

  ros::Subscriber fix_sub;
  // ros::Subscriber sub_Imu_raw;
  ros::Publisher fix_odom_pub;
  ros::Publisher fix_position_pub;
  queue<sensor_msgs::NavSatFix> fix_queue;
  // queue<sensor_msgs::Imu> imu_raw_queue;
  queue<sleipnir_msgs::sensorgps> sleipnir_gps_queue;

  bool init_fix = false;
  double init_fix_odom_x, init_fix_odom_y, init_fix_odom_z; 
  double init_fix_odom_yaw, init_fix_odom_pitch, init_fix_odom_roll;
  Eigen::AngleAxisd rollAngle;
  Eigen::AngleAxisd pitchAngle;
  Eigen::AngleAxisd yawAngle;
  // Eigen::Quaterniond first_imu_raw_att;
  Eigen::Quaterniond init_fix_odom_pose;
  Eigen::Vector3d InitEulerAngle;
  std::string init_fix_zoom;
  double yaw_G2L, pitch_G2L, roll_G2L;
  pcl::PointCloud<PointType>::Ptr GPSHistoryPosition3D;
  bool northp;
  int izone;

  double gps_noise_x, gps_noise_y, gps_noise_z;
  double gps_noise_att;
  bool indoor = false;
  unsigned int buffNum = 0;

  // for KITTI
  // Eigen::Vector3d curr_gps, curr_tfm;
  // Eigen::Vector3d last_gps, last_tfm;
  // std::deque<Eigen::Vector3d> q_gps_err;
  // std::deque<Eigen::Vector3d> q_tfm_err;
  // unsigned int N_gps_err = 10;
  // unsigned int num_gps = 1;
  // double first_threshold = 10.0;
  // double second_threshold = 3.0;
  // Eigen::Vector3d curr_tfm_sum;
  // int aft_abnormal = 0;
  // int set_aft_abnormal = 0;

  // loop detector 
  SCManager scManager;
  float yawDiffRad;
  noiseModel::Base::shared_ptr robustNoiseModel;

  int SCclosestHistoryFrameID; // giseop 
  int RSclosestHistoryFrameID;

  float transformLast[6];
  // lidar在/odom下的位姿
  float transformSum[6];
  float transformIncre[6];
  // T_odom_2_map before optimize
  float transformTobeMapped[6];
  // T_odom_2_map after optimize
  float transformBefMapped[6];
  // lidar在/map下的位姿
  float transformAftMapped[6];

  std::mutex mtx;

  double timeLastProcessing;

  PointType pointOri, pointSel, pointProj, coeff;

  cv::Mat matA0;
  cv::Mat matB0;
  cv::Mat matX0;

  cv::Mat matA1;
  cv::Mat matD1;
  cv::Mat matV1;

  bool isDegenerate;
  cv::Mat matP;

  int laserCloudCornerFromMapDSNum;
  int laserCloudSurfFromMapDSNum;
  int laserCloudCornerLastDSNum;
  int laserCloudSurfLastDSNum;
  int laserCloudOutlierLastDSNum;
  int laserCloudSurfTotalLastDSNum;

  bool potentialLoopFlag;
  double timeSaveFirstCurrentScanForLoopClosure;
  int closestHistoryFrameID;
  int latestFrameIDLoopCloure;

  bool aLoopIsClosed;

  float cRoll, sRoll, cPitch, sPitch, cYaw, sYaw, tX, tY, tZ;
  float ctRoll, stRoll, ctPitch, stPitch, ctYaw, stYaw, tInX, tInY, tInZ;

  // 先验状态 和 后验状态
  gtsam::Rot3 prior_rotation;
  gtsam::Point3 prior_point; 
  gtsam::Pose3 prior_pose;
  gtsam::Vector3 prior_velocity = gtsam::Vector3(0,0,0);
  gtsam::NavState prev_state;
  gtsam::NavState prop_state;
  gtsam::imuBias::ConstantBias prior_imu_bias; // assume zero initial bias
  gtsam::imuBias::ConstantBias prev_bias = prior_imu_bias;

  int imuPointerFront;
  int imuPointerLast;

  double imuTime[imuQueLength_];
  float imuRoll[imuQueLength_];
  float imuPitch[imuQueLength_];
  std::deque<sensor_msgs::Imu> imuQue;

  // IMU预积分的一些参数配置
  // boost::shared_ptr<PreintegratedCombinedMeasurements::Params> imu_params = PreintegratedCombinedMeasurements::Params::MakeSharedD(imuGravity);
  boost::shared_ptr<gtsam::PreintegrationParams> imu_params = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
  // IMU预积分器
  gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;
  // 初始速度噪声模型
  noiseModel::Diagonal::shared_ptr velocity_noise_model = noiseModel::Isotropic::Sigma(3,0.1); // m/s
  noiseModel::Diagonal::shared_ptr bias_noise_model = noiseModel::Isotropic::Sigma(6,1e-3);

  double lastImuTime, curImuTime;

  ofstream fout;
  ofstream fout_evo;
  ofstream fout_gps_evo;
  ofstream fout_init_gps;
  string FileDir;
  string FileName;  

  bool useBetweenFactor = true;
  bool useGPSfactor = true;
  bool customized_gps_msg = true;
  bool isSaveMap = false;
  bool SaveIntensity = false;

  map<int, int> loopIndexContainer; // from new to old

 public:
  MappingHandler(ros::NodeHandle& nh, ros::NodeHandle& pnh) : nh(nh), pnh(pnh) {
    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);

    pnh.param<double>("yaw_G2L", yaw_G2L, 0.03);
    pnh.param<double>("pitch_G2L", pitch_G2L, 0.0);
    pnh.param<double>("roll_G2L", roll_G2L, 0.0);

    pnh.param<bool>("customized_gps_msg", customized_gps_msg, true);
    pnh.param<bool>("SaveMap", isSaveMap, false);
    pnh.param<bool>("SaveIntensity", SaveIntensity, false);

    pnh.param<string>("FileDir", FileDir, "/home/kyle/Downloads/ROSfile/");

    pnh.param<bool>("useBetweenFactor", useBetweenFactor, true);

    pnh.param<bool>("useGPSfactor", useGPSfactor, true);

    pubKeyPoses =
        pnh.advertise<sensor_msgs::PointCloud2>("/key_pose_origin", 2);
    // 当前位姿附近的full点云
    pubLaserCloudSurround =
        pnh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 2);
    // 经map优化的odometry
    pubOdomAftMapped =
        pnh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 5);
    // ???
    pubOdomXYZAftMapped =
        pnh.advertise<nav_msgs::Odometry>("/aft_xyz_mapped_to_init", 5);

    pubLoopConstraintEdge = pnh.advertise<visualization_msgs::MarkerArray>("/loop_closure_constraints", 1);

    // scan context
    subLaserCloudRaw = pnh.subscribe<sensor_msgs::PointCloud2>(
        LIDAR_TOPIC, 2, 
        &MappingHandler::laserCloudRawHandler, this);
    fix_odom_pub = pnh.advertise<nav_msgs::Odometry>("/fix_odom", 5);
    fix_sub = pnh.subscribe("/fix", 50, &MappingHandler::fixHandler, this);

    fix_position_pub = pnh.advertise<sensor_msgs::PointCloud2>("/gps_history_position", 2);

    sleipnir_gps_sub = pnh.subscribe("/sensorgps", 50, &MappingHandler::GpsHancdler, this);

    // sub_livox_msg1 = pnh.subscribe<livox_ros_driver::CustomMsg>("/livox/lidar", 100, &MappingHandler::LivoxMsgCbk1, this);
    
    // 最新的边缘特征点~
    subLaserCloudCornerLast = pnh.subscribe<sensor_msgs::PointCloud2>(
        "/laser_cloud_corner_last", 2,
        &MappingHandler::laserCloudCornerLastHandler, this);
    // 平面特征点
    subLaserCloudSurfLast = pnh.subscribe<sensor_msgs::PointCloud2>(
        "/laser_cloud_surf_last", 2, &MappingHandler::laserCloudSurfLastHandler,
        this);
    // 野点
    subOutlierCloudLast = pnh.subscribe<sensor_msgs::PointCloud2>(
        "/outlier_cloud_last", 2, &MappingHandler::laserCloudOutlierLastHandler,
        this);
    // 前端的里程计
    subLaserOdometry = pnh.subscribe<nav_msgs::Odometry>(
        "/laser_odom_to_init", 5, &MappingHandler::laserOdometryHandler, this);
    // Imu数据
    subImu = pnh.subscribe<sensor_msgs::Imu>(IMU_TOPIC, 50,
                                             &MappingHandler::imuHandler, this);

    pubHistoryKeyFrames =
        pnh.advertise<sensor_msgs::PointCloud2>("/history_cloud", 2);
    // unused
    pubIcpKeyFrames =
        pnh.advertise<sensor_msgs::PointCloud2>("/corrected_cloud", 2);
    pubRecentKeyFrames =
        pnh.advertise<sensor_msgs::PointCloud2>("/recent_cloud", 2);

    downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilterSurf.setLeafSize(0.4, 0.4, 0.4);
    downSizeFilterOutlier.setLeafSize(0.4, 0.4, 0.4);

    downSizeFilterHistoryKeyFrames.setLeafSize(0.4, 0.4, 0.4);
    downSizeFilterSurroundingKeyPoses.setLeafSize(1.0, 1.0, 1.0);

    downSizeFilterGlobalMapKeyPoses.setLeafSize(0.5, 0.5, 0.5);
    downSizeFilterGlobalMapKeyFrames.setLeafSize(0.2, 0.2, 0.2);

    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "aft_mapped";

    aftMappedTrans.frame_id_ = "camera_init";
    aftMappedTrans.child_frame_id_ = "aft_mapped";

    aftMappedXYZTrans.frame_id_ = "map";
    aftMappedXYZTrans.child_frame_id_ = "aft_xyz_mapped";

    allocateMemory();

    FileName = FileDir+"SLAMOdometry.txt";
    fout.open(FileName);
    fout.precision(20);
    fout.clear();

    fout_evo.open(FileDir+"lins_traj_evo.txt");
    fout_evo.precision(15);
    fout_evo.clear();

    fout_gps_evo.open(FileDir+"gps_traj_evo.txt");
    fout_gps_evo.precision(15);
    fout_gps_evo.clear();

    fout_init_gps.open(FileDir+"init_pose.yaml");
    fout_init_gps.precision(15);
    fout_init_gps.clear();
  }

  void allocateMemory() {
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

    kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

    surroundingKeyPoses.reset(new pcl::PointCloud<PointType>());
    surroundingKeyPosesDS.reset(new pcl::PointCloud<PointType>());

    laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());
    laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>());
    laserCloudOutlierLast.reset(new pcl::PointCloud<PointType>());
    laserCloudOutlierLastDS.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfTotalLast.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfTotalLastDS.reset(new pcl::PointCloud<PointType>());

    laserCloudOri.reset(new pcl::PointCloud<PointType>());
    coeffSel.reset(new pcl::PointCloud<PointType>());

    laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
    laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

    kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

    nearHistoryCornerKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
    nearHistoryCornerKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());
    nearHistorySurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
    nearHistorySurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

    latestCornerKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
    latestSurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
    latestSurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

    kdtreeGlobalMap.reset(new pcl::KdTreeFLANN<PointType>());
    globalMapKeyPoses.reset(new pcl::PointCloud<PointType>());
    globalMapKeyPosesDS.reset(new pcl::PointCloud<PointType>());
    globalMapKeyFrames.reset(new pcl::PointCloud<PointType>());
    globalMapKeyFramesDS.reset(new pcl::PointCloud<PointType>());
    laserCloudFullResColor.reset(new pcl::PointCloud<pcl::PointXYZRGB>());

    globalCornerMapKeyFrames.reset(new pcl::PointCloud<PointType>());
    globalSurfMapKeyFrames.reset(new pcl::PointCloud<PointType>());

    // scan context
    RSlatestSurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>()); // giseop
    RSnearHistorySurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
    RSnearHistorySurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

    SClatestSurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
    SCnearHistorySurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
    SCnearHistorySurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());
    GPSHistoryPosition3D.reset(new pcl::PointCloud<PointType>());

    laserCloudRaw.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
    laserCloudRawDS.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
    OriginateGPStoMGRSforTest.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
    // FusiontoUTMforTest.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
    downSizeFilterScancontext.setLeafSize(0.5, 0.5, 0.5);

    timeLaserCloudCornerLast = 0;
    timeLaserCloudSurfLast = 0;
    timeLaserOdometry = 0;
    timeLaserCloudOutlierLast = 0;
    timeLastGloalMapPublish = 0;
    timeLaserCloudRawLast = 0;

    timeLastProcessing = -1;

    newLaserCloudCornerLast = false;
    newLaserCloudSurfLast = false;

    newLaserOdometry = false;
    newLaserCloudOutlierLast = false;
    newLaserCloudRawLast = false;

    for (int i = 0; i < 6; ++i) {
      transformLast[i] = 0;
      transformSum[i] = 0;
      transformIncre[i] = 0;
      transformTobeMapped[i] = 0;
      transformBefMapped[i] = 0;
      transformAftMapped[i] = 0;
    }

    imuPointerFront = 0;
    imuPointerLast = -1;

    for (int i = 0; i < imuQueLength_; ++i) {
      imuTime[i] = 0;
      imuRoll[i] = 0;
      imuPitch[i] = 0;
    }

    gtsam::Vector PriorVector6(6);
    gtsam::Vector OdometryVector6(6);
    // Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;
    // PriorVector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;
    PriorVector6 << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6;
    OdometryVector6 << 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-2;
    priorNoise = noiseModel::Diagonal::Variances(PriorVector6);
    odometryNoise = noiseModel::Diagonal::Variances(OdometryVector6);

    matA0 = cv::Mat(5, 3, CV_32F, cv::Scalar::all(0));
    matB0 = cv::Mat(5, 1, CV_32F, cv::Scalar::all(-1));
    matX0 = cv::Mat(3, 1, CV_32F, cv::Scalar::all(0));

    matA1 = cv::Mat(3, 3, CV_32F, cv::Scalar::all(0));
    matD1 = cv::Mat(1, 3, CV_32F, cv::Scalar::all(0));
    matV1 = cv::Mat(3, 3, CV_32F, cv::Scalar::all(0));

    isDegenerate = false;
    matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));

    laserCloudCornerFromMapDSNum = 0;
    laserCloudSurfFromMapDSNum = 0;
    laserCloudCornerLastDSNum = 0;
    laserCloudSurfLastDSNum = 0;
    laserCloudOutlierLastDSNum = 0;
    laserCloudSurfTotalLastDSNum = 0;

    potentialLoopFlag = false;
    aLoopIsClosed = false;

    latestFrameID = 0;

    // 配置噪声协方差矩阵
    Matrix33 measured_acc_cov = I_3x3 * pow(accel_noise_sigma, 2);
    Matrix33 measured_omega_cov = I_3x3 * pow(gyro_noise_sigma, 2);
    Matrix33 integration_error_cov =
        I_3x3 * 1e-8;  // error committed in integrating position from velocities

    // PreintegrationBase params:
    imu_params->accelerometerCovariance =
        measured_acc_cov;  // acc white noise in continuous
    imu_params->integrationCovariance =
        integration_error_cov;  // integration uncertainty continuous
    // should be using 2nd order integration
    // PreintegratedRotation params:
    imu_params->gyroscopeCovariance =
        measured_omega_cov;  // gyro white noise in continuous

    imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(imu_params, prior_imu_bias);

    imuIntegratorImu_->resetIntegrationAndSetBias(prior_imu_bias);
  }

  // transformBefMapped -> transformTobeMapped
  // odom -> map
  // 基于匀速模型，根据上次微调的结果和odometry这次与上次计算的结果，
  // 猜测一个新的世界坐标系的转换矩阵transformTobeMapped[6]
  // 能不能换成四元数表达???
  void transformAssociateToMap() {
    float x1 =
        cos(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) -
        sin(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);
    float y1 = transformBefMapped[4] - transformSum[4];
    float z1 =
        sin(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) +
        cos(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);

    float x2 = x1;
    float y2 = cos(transformSum[0]) * y1 + sin(transformSum[0]) * z1;
    float z2 = -sin(transformSum[0]) * y1 + cos(transformSum[0]) * z1;

    transformIncre[3] = cos(transformSum[2]) * x2 + sin(transformSum[2]) * y2;
    transformIncre[4] = -sin(transformSum[2]) * x2 + cos(transformSum[2]) * y2;
    transformIncre[5] = z2;

    float sbcx = sin(transformSum[0]);
    float cbcx = cos(transformSum[0]);
    float sbcy = sin(transformSum[1]);
    float cbcy = cos(transformSum[1]);
    float sbcz = sin(transformSum[2]);
    float cbcz = cos(transformSum[2]);

    float sblx = sin(transformBefMapped[0]);
    float cblx = cos(transformBefMapped[0]);
    float sbly = sin(transformBefMapped[1]);
    float cbly = cos(transformBefMapped[1]);
    float sblz = sin(transformBefMapped[2]);
    float cblz = cos(transformBefMapped[2]);

    float salx = sin(transformAftMapped[0]);
    float calx = cos(transformAftMapped[0]);
    float saly = sin(transformAftMapped[1]);
    float caly = cos(transformAftMapped[1]);
    float salz = sin(transformAftMapped[2]);
    float calz = cos(transformAftMapped[2]);

    float srx = -sbcx * (salx * sblx + calx * cblx * salz * sblz +
                         calx * calz * cblx * cblz) -
                cbcx * sbcy *
                    (calx * calz * (cbly * sblz - cblz * sblx * sbly) -
                     calx * salz * (cbly * cblz + sblx * sbly * sblz) +
                     cblx * salx * sbly) -
                cbcx * cbcy *
                    (calx * salz * (cblz * sbly - cbly * sblx * sblz) -
                     calx * calz * (sbly * sblz + cbly * cblz * sblx) +
                     cblx * cbly * salx);
    transformTobeMapped[0] = -asin(srx);

    float srycrx = sbcx * (cblx * cblz * (caly * salz - calz * salx * saly) -
                           cblx * sblz * (caly * calz + salx * saly * salz) +
                           calx * saly * sblx) -
                   cbcx * cbcy *
                       ((caly * calz + salx * saly * salz) *
                            (cblz * sbly - cbly * sblx * sblz) +
                        (caly * salz - calz * salx * saly) *
                            (sbly * sblz + cbly * cblz * sblx) -
                        calx * cblx * cbly * saly) +
                   cbcx * sbcy *
                       ((caly * calz + salx * saly * salz) *
                            (cbly * cblz + sblx * sbly * sblz) +
                        (caly * salz - calz * salx * saly) *
                            (cbly * sblz - cblz * sblx * sbly) +
                        calx * cblx * saly * sbly);
    float crycrx = sbcx * (cblx * sblz * (calz * saly - caly * salx * salz) -
                           cblx * cblz * (saly * salz + caly * calz * salx) +
                           calx * caly * sblx) +
                   cbcx * cbcy *
                       ((saly * salz + caly * calz * salx) *
                            (sbly * sblz + cbly * cblz * sblx) +
                        (calz * saly - caly * salx * salz) *
                            (cblz * sbly - cbly * sblx * sblz) +
                        calx * caly * cblx * cbly) -
                   cbcx * sbcy *
                       ((saly * salz + caly * calz * salx) *
                            (cbly * sblz - cblz * sblx * sbly) +
                        (calz * saly - caly * salx * salz) *
                            (cbly * cblz + sblx * sbly * sblz) -
                        calx * caly * cblx * sbly);
    transformTobeMapped[1] = atan2(srycrx / cos(transformTobeMapped[0]),
                                   crycrx / cos(transformTobeMapped[0]));

    float srzcrx = (cbcz * sbcy - cbcy * sbcx * sbcz) *
                       (calx * salz * (cblz * sbly - cbly * sblx * sblz) -
                        calx * calz * (sbly * sblz + cbly * cblz * sblx) +
                        cblx * cbly * salx) -
                   (cbcy * cbcz + sbcx * sbcy * sbcz) *
                       (calx * calz * (cbly * sblz - cblz * sblx * sbly) -
                        calx * salz * (cbly * cblz + sblx * sbly * sblz) +
                        cblx * salx * sbly) +
                   cbcx * sbcz *
                       (salx * sblx + calx * cblx * salz * sblz +
                        calx * calz * cblx * cblz);
    float crzcrx = (cbcy * sbcz - cbcz * sbcx * sbcy) *
                       (calx * calz * (cbly * sblz - cblz * sblx * sbly) -
                        calx * salz * (cbly * cblz + sblx * sbly * sblz) +
                        cblx * salx * sbly) -
                   (sbcy * sbcz + cbcy * cbcz * sbcx) *
                       (calx * salz * (cblz * sbly - cbly * sblx * sblz) -
                        calx * calz * (sbly * sblz + cbly * cblz * sblx) +
                        cblx * cbly * salx) +
                   cbcx * cbcz *
                       (salx * sblx + calx * cblx * salz * sblz +
                        calx * calz * cblx * cblz);
    transformTobeMapped[2] = atan2(srzcrx / cos(transformTobeMapped[0]),
                                   crzcrx / cos(transformTobeMapped[0]));

    x1 = cos(transformTobeMapped[2]) * transformIncre[3] -
         sin(transformTobeMapped[2]) * transformIncre[4];
    y1 = sin(transformTobeMapped[2]) * transformIncre[3] +
         cos(transformTobeMapped[2]) * transformIncre[4];
    z1 = transformIncre[5];

    x2 = x1;
    y2 = cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
    z2 = sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

    transformTobeMapped[3] =
        transformAftMapped[3] -
        (cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2);
    transformTobeMapped[4] = transformAftMapped[4] - y2;
    transformTobeMapped[5] =
        transformAftMapped[5] -
        (-sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2);
  }

  // 更新T_odom_2_map
  void transformUpdate() {
    // 如果接收到IMU
    if (imuPointerLast >= 0) {
      float imuRollLast = 0, imuPitchLast = 0;
      // 搜索最新的一帧IMU
      while (imuPointerFront != imuPointerLast) {
        if (timeLaserOdometry + SCAN_PERIOD < imuTime[imuPointerFront]) {
          break;
        }
        imuPointerFront = (imuPointerFront + 1) % imuQueLength_; // 提取最新一帧的IMU的索引
      }

      if (timeLaserOdometry + SCAN_PERIOD > imuTime[imuPointerFront]) { // 提取最新一帧IMU的roll 和 pitch
        imuRollLast = imuRoll[imuPointerFront];
        imuPitchLast = imuPitch[imuPointerFront];
      } else { // 线性差值获取
        int imuPointerBack =
            (imuPointerFront + imuQueLength_ - 1) % imuQueLength_;
        float ratioFront =
            (timeLaserOdometry + SCAN_PERIOD - imuTime[imuPointerBack]) /
            (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
        float ratioBack =
            (imuTime[imuPointerFront] - timeLaserOdometry - SCAN_PERIOD) /
            (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

        imuRollLast = imuRoll[imuPointerFront] * ratioFront +
                      imuRoll[imuPointerBack] * ratioBack;
        imuPitchLast = imuPitch[imuPointerFront] * ratioFront +
                       imuPitch[imuPointerBack] * ratioBack;
      }

      transformTobeMapped[0] =
          0.998 * transformTobeMapped[0] + 0.002 * imuPitchLast;
      transformTobeMapped[2] =
          0.998 * transformTobeMapped[2] + 0.002 * imuRollLast;
    }

    for (int i = 0; i < 6; i++) {
      transformBefMapped[i] = transformSum[i];
      transformAftMapped[i] = transformTobeMapped[i];
    }
  }

  void updatePointAssociateToMapSinCos() {
    cRoll = cos(transformTobeMapped[0]);
    sRoll = sin(transformTobeMapped[0]);

    cPitch = cos(transformTobeMapped[1]);
    sPitch = sin(transformTobeMapped[1]);

    cYaw = cos(transformTobeMapped[2]);
    sYaw = sin(transformTobeMapped[2]);

    tX = transformTobeMapped[3];
    tY = transformTobeMapped[4];
    tZ = transformTobeMapped[5];
  }

  // 将lidar坐标系下的点转换到/map坐标系下
  void pointAssociateToMap(PointType const* const pi, PointType* const po) {
    float x1 = cYaw * pi->x - sYaw * pi->y;
    float y1 = sYaw * pi->x + cYaw * pi->y;
    float z1 = pi->z;

    float x2 = x1;
    float y2 = cRoll * y1 - sRoll * z1;
    float z2 = sRoll * y1 + cRoll * z1;

    po->x = cPitch * x2 + sPitch * z2 + tX;
    po->y = y2 + tY;
    po->z = -sPitch * x2 + cPitch * z2 + tZ;
    po->intensity = pi->intensity;
  }

  void updateTransformPointCloudSinCos(PointTypePose* tIn) {
    ctRoll = cos(tIn->roll);
    stRoll = sin(tIn->roll);

    ctPitch = cos(tIn->pitch);
    stPitch = sin(tIn->pitch);

    ctYaw = cos(tIn->yaw);
    stYaw = sin(tIn->yaw);

    tInX = tIn->x;
    tInY = tIn->y;
    tInZ = tIn->z;
  }

  pcl::PointCloud<PointType>::Ptr transformPointCloud(
      pcl::PointCloud<PointType>::Ptr cloudIn) {
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    PointType* pointFrom;
    PointType pointTo;

    int cloudSize = cloudIn->points.size();
    cloudOut->resize(cloudSize);

    for (int i = 0; i < cloudSize; ++i) {
      pointFrom = &cloudIn->points[i];
      float x1 = ctYaw * pointFrom->x - stYaw * pointFrom->y;
      float y1 = stYaw * pointFrom->x + ctYaw * pointFrom->y;
      float z1 = pointFrom->z;

      float x2 = x1;
      float y2 = ctRoll * y1 - stRoll * z1;
      float z2 = stRoll * y1 + ctRoll * z1;

      pointTo.x = ctPitch * x2 + stPitch * z2 + tInX;
      pointTo.y = y2 + tInY;
      pointTo.z = -stPitch * x2 + ctPitch * z2 + tInZ;
      pointTo.intensity = pointFrom->intensity;

      cloudOut->points[i] = pointTo;
    }
    return cloudOut;
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformPointCloud(
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudIn, PointTypePose* transformIn) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudOut(new pcl::PointCloud<pcl::PointXYZRGB>());

    pcl::PointXYZRGB* pointFrom;
    pcl::PointXYZRGB pointTo;

    int cloudSize = cloudIn->points.size();
    cloudOut->resize(cloudSize);

    for (int i = 0; i < cloudSize; ++i) {
      pointFrom = &cloudIn->points[i];
      float x1 = cos(transformIn->yaw) * pointFrom->x -
                 sin(transformIn->yaw) * pointFrom->y;
      float y1 = sin(transformIn->yaw) * pointFrom->x +
                 cos(transformIn->yaw) * pointFrom->y;
      float z1 = pointFrom->z;

      float x2 = x1;
      float y2 = cos(transformIn->roll) * y1 - sin(transformIn->roll) * z1;
      float z2 = sin(transformIn->roll) * y1 + cos(transformIn->roll) * z1;

      pointTo.x = cos(transformIn->pitch) * x2 + sin(transformIn->pitch) * z2 +
                  transformIn->x;
      pointTo.y = y2 + transformIn->y;
      pointTo.z = -sin(transformIn->pitch) * x2 + cos(transformIn->pitch) * z2 +
                  transformIn->z;
      // pointTo.intensity = pointFrom->intensity;
      pointTo.r = pointFrom->r;
      pointTo.g = pointFrom->g;
      pointTo.b = pointFrom->b;

      cloudOut->points[i] = pointTo;
    }
    return cloudOut;
  }

  pcl::PointCloud<PointType>::Ptr transformPointCloud(
      pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn) {
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    PointType* pointFrom;
    PointType pointTo;

    int cloudSize = cloudIn->points.size();
    cloudOut->resize(cloudSize);

    for (int i = 0; i < cloudSize; ++i) {
      pointFrom = &cloudIn->points[i];
      float x1 = cos(transformIn->yaw) * pointFrom->x -
                 sin(transformIn->yaw) * pointFrom->y;
      float y1 = sin(transformIn->yaw) * pointFrom->x +
                 cos(transformIn->yaw) * pointFrom->y;
      float z1 = pointFrom->z;

      float x2 = x1;
      float y2 = cos(transformIn->roll) * y1 - sin(transformIn->roll) * z1;
      float z2 = sin(transformIn->roll) * y1 + cos(transformIn->roll) * z1;

      pointTo.x = cos(transformIn->pitch) * x2 + sin(transformIn->pitch) * z2 +
                  transformIn->x;
      pointTo.y = y2 + transformIn->y;
      pointTo.z = -sin(transformIn->pitch) * x2 + cos(transformIn->pitch) * z2 +
                  transformIn->z;
      pointTo.intensity = pointFrom->intensity;

      cloudOut->points[i] = pointTo;
    }
    return cloudOut;
  }

  void RGBpointAssociateToMap(PointType const *const pi,
                            pcl::PointXYZRGB *const po) {
      // Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
      // Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
      po->x = pi->x;
      po->y = pi->y;
      po->z = pi->z;
      int reflection_map = pi->intensity;
      if (reflection_map < 30) {
        int green = (reflection_map * 255 / 30);
        po->r = 0;
        po->g = green & 0xff;
        po->b = 0xff;
      } else if (reflection_map < 90) {
        int blue = (((90 - reflection_map) * 255) / 60);
        po->r = 0x0;
        po->g = 0xff;
        po->b = blue & 0xff;
      } else if (reflection_map < 150) {
        int red = ((reflection_map - 90) * 255 / 60);
        po->r = red & 0xff;
        po->g = 0xff;
        po->b = 0x0;
      } else {
        int green = (((255 - reflection_map) * 255) / (255 - 150));
        po->r = 0xff;
        po->g = green & 0xff;
        po->b = 0;
      }
  }

//   scan context
  void laserCloudRawHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudRawLast = msg->header.stamp.toSec();
        laserCloudRaw->clear();
        pcl::fromROSMsg(*msg, *laserCloudRaw);
        newLaserCloudRawLast = true;
    }

  void fixHandler(const sensor_msgs::NavSatFix::ConstPtr& msg)
  {
    if (msg->status.status == sensor_msgs::NavSatStatus::STATUS_NO_FIX) {
      ROS_DEBUG_THROTTLE(60,"No fix.");
      return;
    }

    if (msg->header.stamp == ros::Time(0)) {
      return;
    }

    fix_queue.push(*msg);
  }

  // sleipnir GPS
  void GpsHancdler(const sleipnir_msgs::sensorgps::ConstPtr &msg)
  {
    sleipnir_msgs::sensorgps currGps = *msg;
    // currGps.header.stamp.sec -= 18;
    sleipnir_gps_queue.push(currGps);

    // cout<<"receive currGps"<<endl;
  }

  void laserCloudOutlierLastHandler(
      const sensor_msgs::PointCloud2ConstPtr& msg) {
    timeLaserCloudOutlierLast = msg->header.stamp.toSec();
    laserCloudOutlierLast->clear();
    pcl::fromROSMsg(*msg, *laserCloudOutlierLast);
    newLaserCloudOutlierLast = true;
  }

  void laserCloudCornerLastHandler(
      const sensor_msgs::PointCloud2ConstPtr& msg) {
    timeLaserCloudCornerLast = msg->header.stamp.toSec();
    laserCloudCornerLast->clear();
    pcl::fromROSMsg(*msg, *laserCloudCornerLast);
    newLaserCloudCornerLast = true;
  }

  void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg) {
    timeLaserCloudSurfLast = msg->header.stamp.toSec();
    laserCloudSurfLast->clear();
    pcl::fromROSMsg(*msg, *laserCloudSurfLast);
    newLaserCloudSurfLast = true;
  }

  void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr& laserOdometry) {
    timeLaserOdometry = laserOdometry->header.stamp.toSec();
    double roll, pitch, yaw;
    geometry_msgs::Quaternion geoQuat = laserOdometry->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w))
        .getRPY(roll, pitch, yaw);
    transformSum[0] = -pitch;
    transformSum[1] = -yaw;
    transformSum[2] = roll;
    transformSum[3] = laserOdometry->pose.pose.position.x;
    transformSum[4] = laserOdometry->pose.pose.position.y;
    transformSum[5] = laserOdometry->pose.pose.position.z;
    newLaserOdometry = true;
  }

  // 只提取imu中的roll, pitch???
  void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn) {
    std::lock_guard<std::mutex> lock(mtx);
    double roll, pitch, yaw;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(imuIn->orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
    imuPointerLast = (imuPointerLast + 1) % imuQueLength_;
    imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
    imuRoll[imuPointerLast] = roll;
    imuPitch[imuPointerLast] = pitch;

    sensor_msgs::Imu thisImu = *imuIn;
    imuQue.push_back(thisImu);
  }

  sleipnir_msgs::sensorgps GpsSlerp(const sleipnir_msgs::sensorgps forGps, const sleipnir_msgs::sensorgps endGps, double timestamp_)
  {
    // cout<<"forGps: "<<forGps.header.stamp<<endl;
    // cout<<"endGps: "<<endGps.header.stamp<<endl;
    // cout<<"timestamp_: "<<timestamp_<<endl;
    double coefficient = (timestamp_ - forGps.header.stamp.toSec())/(endGps.header.stamp.toSec()-forGps.header.stamp.toSec());
    // cout<<"coefficient: "<<coefficient<<endl;
    sleipnir_msgs::sensorgps resGps;
    resGps.header.stamp = ros::Time().fromSec(timestamp_);

    resGps.lat = forGps.lat + (endGps.lat - forGps.lat)*coefficient;
    resGps.lon = forGps.lon + (endGps.lon - forGps.lon)*coefficient;
    resGps.alt = forGps.alt + (endGps.alt - forGps.alt)*coefficient;

    if(abs(endGps.heading-forGps.heading)<=180)
    {
      resGps.heading = forGps.heading + (endGps.heading - forGps.heading)*coefficient;
    }
    else if(endGps.heading>forGps.heading)
    {
      resGps.heading = forGps.heading + (endGps.heading-360.0 - forGps.heading)*coefficient;
    }
    else if(endGps.heading<forGps.heading)
    {
      resGps.heading = forGps.heading + (endGps.heading+360.0 - forGps.heading)*coefficient;
    }
    else
    {
      resGps.heading = endGps.heading;
    }
    resGps.pitch = forGps.pitch + (endGps.pitch - forGps.pitch)*coefficient;
    resGps.roll = forGps.roll + (endGps.roll - forGps.roll)*coefficient;
    // resGps.lat = endGps.lat;
    // resGps.lon = endGps.lon;
    // resGps.alt = endGps.alt;
    // resGps.heading = endGps.heading;
    // resGps.pitch = endGps.pitch;
    // resGps.roll = endGps.roll;
    resGps.satenum = endGps.satenum;
    resGps.status = endGps.status;
    resGps.velocity = endGps.velocity;
    resGps.x = endGps.x;
    resGps.y = endGps.y;

    return resGps;
  }

  // // 创建一个循环队列用于存储雷达帧
  // uint64_t TO_MERGE_CNT = 1; 
  // // bool b_dbg_line = false;
  // std::vector<livox_ros_driver::CustomMsgConstPtr> livox_data;
  // void LivoxMsgCbk1(const livox_ros_driver::CustomMsgConstPtr& livox_msg_in) {
  //   // cout<<"recevie a livox msg!"<<endl;
  //   livox_data.push_back(livox_msg_in);
  //   // 第一帧则跳过
  //   if (livox_data.size() < TO_MERGE_CNT) return;

  //   pcl::PointCloud<pcl::PointXYZINormal> pcl_in;

  //   for (size_t j = 0; j < livox_data.size(); j++) {
  //     // 通过引用，方便操作每一帧
  //     auto& livox_msg = livox_data[j];
  //     // 获取该帧最后一个点的相对时间
  //     auto time_end = livox_msg->points.back().offset_time;
  //     // 重新组织成PCL的点云
  //     for (unsigned int i = 0; i < livox_msg->point_num; ++i) {
  //       pcl::PointXYZINormal pt;
  //       pt.x = livox_msg->points[i].x;
  //       pt.y = livox_msg->points[i].y;
  //       pt.z = livox_msg->points[i].z;
  // //      if (pt.z < -0.3) continue; // delete some outliers (our Horizon's assembly height is 0.3 meters)
  //       float s = livox_msg->points[i].offset_time / (float)time_end;
  // //       ROS_INFO("_s-------- %.6f ",s);
  //       // 线数存在整数部分，时间偏移存在
  //       pt.intensity = livox_msg->points[i].line + s*0.1; // The integer part is line number and the decimal part is timestamp
  // //      ROS_INFO("intensity-------- %.6f ",pt.intensity);
  //       pt.curvature = livox_msg->points[i].reflectivity * 0.1;
  //       // ROS_INFO("pt.curvature-------- %.3f ",pt.curvature);
  //       pcl_in.push_back(pt);
  //     }
  //   }

  //   sensor_msgs::PointCloud2 msg;
  //   pcl::toROSMsg(pcl_in, msg);
  //   msg.header.stamp = livox_msg_in->header.stamp;
  //   msg.header.frame_id = "livox";

  //   // std::lock_guard<std::mutex> lock(mtx);
  //   livox_points_msgs.push(msg);
  //   // mtx.unlock();
  //   livox_data.clear();
  // }

  // camera : camera_init -> /aft_map
  void publishTF() {
    if (isnan(transformAftMapped[0])) transformAftMapped[0] = 0.0;
    if (isnan(transformAftMapped[1])) transformAftMapped[1] = 0.0;
    if (isnan(transformAftMapped[2])) transformAftMapped[2] = 0.0;
    if (isnan(transformAftMapped[3])) transformAftMapped[3] = 0.0;
    if (isnan(transformAftMapped[4])) transformAftMapped[4] = 0.0;
    if (isnan(transformAftMapped[5])) transformAftMapped[5] = 0.0;

    if (isnan(transformBefMapped[0])) transformBefMapped[0] = 0.0;
    if (isnan(transformBefMapped[1])) transformBefMapped[1] = 0.0;
    if (isnan(transformBefMapped[2])) transformBefMapped[2] = 0.0;
    if (isnan(transformBefMapped[3])) transformBefMapped[3] = 0.0;
    if (isnan(transformBefMapped[4])) transformBefMapped[4] = 0.0;
    if (isnan(transformBefMapped[5])) transformBefMapped[5] = 0.0;

    // @TODO
    geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(
        transformAftMapped[2], -transformAftMapped[0], -transformAftMapped[1]);

    odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
    odomAftMapped.pose.pose.orientation.x = -geoQuat.y;
    odomAftMapped.pose.pose.orientation.y = -geoQuat.z;
    odomAftMapped.pose.pose.orientation.z = geoQuat.x;
    odomAftMapped.pose.pose.orientation.w = geoQuat.w;
    odomAftMapped.pose.pose.position.x = transformAftMapped[3];
    odomAftMapped.pose.pose.position.y = transformAftMapped[4];
    odomAftMapped.pose.pose.position.z = transformAftMapped[5];
    odomAftMapped.twist.twist.angular.x = transformBefMapped[0];
    odomAftMapped.twist.twist.angular.y = transformBefMapped[1];
    odomAftMapped.twist.twist.angular.z = transformBefMapped[2];
    odomAftMapped.twist.twist.linear.x = transformBefMapped[3];
    odomAftMapped.twist.twist.linear.y = transformBefMapped[4];
    odomAftMapped.twist.twist.linear.z = transformBefMapped[5];
    pubOdomAftMapped.publish(odomAftMapped);

    aftMappedTrans.stamp_ = ros::Time().fromSec(timeLaserOdometry);
    aftMappedTrans.setRotation(
        tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
    aftMappedTrans.setOrigin(tf::Vector3(
        transformAftMapped[3], transformAftMapped[4], transformAftMapped[5]));
    tfBroadcaster.sendTransform(aftMappedTrans);
  }

  void publishXYZTF() {
    geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(
        transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]);

    odomXYZAftMapped.header.frame_id = "map";
    odomXYZAftMapped.child_frame_id = "aft_xyz_mapped";

    odomXYZAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
    odomXYZAftMapped.pose.pose.orientation.x = geoQuat.x;
    odomXYZAftMapped.pose.pose.orientation.y = geoQuat.y;
    odomXYZAftMapped.pose.pose.orientation.z = geoQuat.z;
    odomXYZAftMapped.pose.pose.orientation.w = geoQuat.w;
    odomXYZAftMapped.pose.pose.position.x = transformAftMapped[5];
    odomXYZAftMapped.pose.pose.position.y = transformAftMapped[3];
    odomXYZAftMapped.pose.pose.position.z =
        transformAftMapped[4];  //-transformAftMapped[4]
    odomXYZAftMapped.twist.twist.angular.x = transformBefMapped[2];
    odomXYZAftMapped.twist.twist.angular.y = transformBefMapped[0];
    odomXYZAftMapped.twist.twist.angular.z = transformBefMapped[1];
    odomXYZAftMapped.twist.twist.linear.x = transformBefMapped[5];
    odomXYZAftMapped.twist.twist.linear.y = transformBefMapped[3];
    odomXYZAftMapped.twist.twist.linear.z =
        transformBefMapped[4];  //-transformBefMapped[4]
    pubOdomXYZAftMapped.publish(odomXYZAftMapped);

    aftMappedXYZTrans.stamp_ = ros::Time().fromSec(timeLaserOdometry);
    aftMappedXYZTrans.setRotation(
        tf::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w));
    aftMappedXYZTrans.setOrigin(
        tf::Vector3(transformAftMapped[5], transformAftMapped[3],
                    transformAftMapped[4]));  //-transformAftMapped[4]
    tfXYZBroadcaster.sendTransform(aftMappedXYZTrans);

  }

  void publishKeyPosesAndFrames() {
    if (pubKeyPoses.getNumSubscribers() != 0) {
      sensor_msgs::PointCloud2 cloudMsgTemp;
      pcl::toROSMsg(*cloudKeyPoses3D, cloudMsgTemp);
      cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
      cloudMsgTemp.header.frame_id = "camera_init";
      pubKeyPoses.publish(cloudMsgTemp);
    }

    if (pubRecentKeyFrames.getNumSubscribers() != 0) {
      sensor_msgs::PointCloud2 cloudMsgTemp;
      pcl::toROSMsg(*laserCloudSurfFromMapDS, cloudMsgTemp);
      cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
      cloudMsgTemp.header.frame_id = "camera_init";
      pubRecentKeyFrames.publish(cloudMsgTemp);
    }
  }

  void visualizeGlobalMapThread() {
    ros::Rate rate(0.2);
    while (ros::ok()) {
      rate.sleep();
      publishGlobalMap();
    }

    // save mapping odometry
    cout<<"cloudKeyPoses6D: "<<cloudKeyPoses6D->size()<<endl;
    Eigen::Matrix<double, 6, 1> thisPose;
    Eigen::Affine3d T_thisPose = Eigen::Affine3d::Identity();
    size_t poseCount = cloudKeyPoses6D->size();
    for(size_t i=0;i<poseCount;i++)
    {
      fout<<cloudKeyPoses6D->points[i].time * 1e9<<endl;
      thisPose[0] = cloudKeyPoses6D->points[i].z;
      thisPose[1] = cloudKeyPoses6D->points[i].x;
      thisPose[2] = cloudKeyPoses6D->points[i].y;
      thisPose[3] = cloudKeyPoses6D->points[i].yaw;
      thisPose[4] = cloudKeyPoses6D->points[i].roll;
      thisPose[5] = cloudKeyPoses6D->points[i].pitch;

      Eigen::AngleAxisd roll = Eigen::AngleAxisd(thisPose[3], Eigen::Vector3d::UnitX());
      Eigen::AngleAxisd pitch = Eigen::AngleAxisd(thisPose[4], Eigen::Vector3d::UnitY());
      Eigen::AngleAxisd yaw = Eigen::AngleAxisd(thisPose[5], Eigen::Vector3d::UnitZ());
      T_thisPose.rotate((yaw*roll*pitch).toRotationMatrix());
      T_thisPose.pretranslate(thisPose.block<3,1>(0,0));

      fout<<T_thisPose.matrix()<<endl;

      Eigen::Quaterniond temp_q(T_thisPose.rotation());
      fout_evo<<cloudKeyPoses6D->points[i].time<<" "
              <<T_thisPose.translation().x()<<" "
              <<T_thisPose.translation().y()<<" "
              <<T_thisPose.translation().z()<<" "
              <<temp_q.x()<<" "
              <<temp_q.y()<<" "
              <<temp_q.z()<<" "
              <<temp_q.w()<<endl;

      T_thisPose.setIdentity();
    }
    fout_evo.close();
    fout.close();
    fout_gps_evo.close();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transposeGlobalPcl(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointXYZRGB tempPoint;
    size_t PclSize = laserCloudFullResColor->size();
    cout<<"PclSize: "<<PclSize<<endl;
    for(size_t i=0; i<PclSize; i++)
    {
      tempPoint = laserCloudFullResColor->points[i];
      laserCloudFullResColor->points[i].x = tempPoint.z;
      laserCloudFullResColor->points[i].y = tempPoint.x;
      laserCloudFullResColor->points[i].z = tempPoint.y;

      // globalMapKeyFramesDS->points[i].intensity = (globalMapKeyFramesDS->points[i].intensity>10?100:globalMapKeyFramesDS->points[i].intensity);
    }

    PointTypePose InitPose;
    // 为了采集综合楼3楼的数据所以注释掉
    //UTM->MGRS
    string mgrs;
    // int prec=5, 意思是精确到m，如果prec=7，精确到cm，如此类推
    MGRS::Forward(izone, northp, init_fix_odom_x, init_fix_odom_y, 7, mgrs);

    InitPose.x = stod(mgrs.substr(5, 7).insert(5, "."));
    InitPose.y = stod(mgrs.substr(12).insert(5, "."));
    InitPose.yaw = init_fix_odom_yaw;
    // cout<<setprecision(15)<<"InitPose: "<<InitPose.x<<", "<<InitPose.y<<", "<<InitPose.yaw<<endl;
    // *transposeGlobalPcl = *transformPointCloud(laserCloudFullResColor, &InitPose);
    fout_init_gps<<"# 建图起始点的MGRS坐标，用于静态变换点云地图到MGRS坐标系下"<<endl;
    fout_init_gps<<"InitPose_x: "<<InitPose.x<<endl;
    fout_init_gps<<"InitPose_y: "<<InitPose.y<<endl;
    fout_init_gps<<"InitPose_yaw: "<<InitPose.yaw<<endl;
    fout_init_gps.close();

    if(isSaveMap)
    {
      pcl::io::savePCDFileBinary (FileDir+"CornerMap.pcd", *globalCornerMapKeyFrames);
      pcl::io::savePCDFileBinary (FileDir+"SurfMap.pcd", *globalSurfMapKeyFrames);
      // pcl::io::savePCDFileASCII ("/home/kyle/ros/kyle_ws/src/lins-gps-iris/pcd/FullMap.pcd", *laserCloudFullResColor);
      // pcl::io::savePCDFileASCII ("/home/kyle/ros/kyle_ws/src/lins-gps-iris/pcd/originateGPStoUTM.pcd", *OriginateGPStoMGRSforTest);
      // pcl::io::savePCDFileASCII ("/home/kyle/ros/kyle_ws/src/lins-gps-iris/pcd/FusiontoUTM.pcd", *FusiontoUTMforTest);
    }
    globalCornerMapKeyFrames->clear();
    globalSurfMapKeyFrames->clear();
    globalMapKeyFramesDS->clear();
  }

  void visualizeLoopClosure()
  {
      visualization_msgs::MarkerArray markerArray;
      // loop nodes
      visualization_msgs::Marker markerNode;
      markerNode.header.frame_id = "camera_init";
      markerNode.header.stamp = ros::Time().fromSec(timeLaserOdometry);
      markerNode.action = visualization_msgs::Marker::ADD;
      markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
      markerNode.ns = "loop_nodes";
      markerNode.id = 0;
      markerNode.pose.orientation.w = 1;
      markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
      markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
      markerNode.color.a = 1;
      // loop edges
      visualization_msgs::Marker markerEdge;
      markerEdge.header.frame_id = "camera_init";
      markerEdge.header.stamp = ros::Time().fromSec(timeLaserOdometry);
      markerEdge.action = visualization_msgs::Marker::ADD;
      markerEdge.type = visualization_msgs::Marker::LINE_LIST;
      markerEdge.ns = "loop_edges";
      markerEdge.id = 1;
      markerEdge.pose.orientation.w = 1;
      markerEdge.scale.x = 0.1; markerEdge.scale.y = 0.1; markerEdge.scale.z = 0.1;
      markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
      markerEdge.color.a = 1;

      for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
      {
          int key_cur = it->first;
          int key_pre = it->second;
          geometry_msgs::Point p;
          p.x = cloudKeyPoses6D->points[key_cur].x;
          p.y = cloudKeyPoses6D->points[key_cur].y;
          p.z = cloudKeyPoses6D->points[key_cur].z;
          markerNode.points.push_back(p);
          markerEdge.points.push_back(p);
          p.x = cloudKeyPoses6D->points[key_pre].x;
          p.y = cloudKeyPoses6D->points[key_pre].y;
          p.z = cloudKeyPoses6D->points[key_pre].z;
          markerNode.points.push_back(p);
          markerEdge.points.push_back(p);
      }

      markerArray.markers.push_back(markerNode);
      markerArray.markers.push_back(markerEdge);
      pubLoopConstraintEdge.publish(markerArray);
  }

  void publishGlobalMap() {
    if (pubLaserCloudSurround.getNumSubscribers() == 0) return;

    if (cloudKeyPoses3D->points.empty() == true) return;

    std::vector<int> pointSearchIndGlobalMap;
    std::vector<float> pointSearchSqDisGlobalMap;

    globalCornerMapKeyFrames->clear();
    globalSurfMapKeyFrames->clear();
    globalMapKeyFramesDS->clear();

    mtx.lock();
    kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
    kdtreeGlobalMap->radiusSearch(
        currentRobotPosPoint, globalMapVisualizationSearchRadius,
        pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
    mtx.unlock();

    for (int i = 0; i < pointSearchIndGlobalMap.size(); ++i)
      globalMapKeyPoses->points.push_back(
          cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);

    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

    for (int i = 0; i < globalMapKeyPosesDS->points.size(); ++i) {
      int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
      *globalCornerMapKeyFrames +=
          *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],
                               &cloudKeyPoses6D->points[thisKeyInd]);
      *globalSurfMapKeyFrames += *transformPointCloud(
          surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
      *globalSurfMapKeyFrames +=
          *transformPointCloud(outlierCloudKeyFrames[thisKeyInd],
                               &cloudKeyPoses6D->points[thisKeyInd]);

      // 保存含强度信息的地图，但是会比较稠密
      // 这里出问题了，如果检测到回环后，这些点云始终没有被矫正
      if(SaveIntensity)
      {
        *globalMapKeyFrames += *originCloudKeyFrames[thisKeyInd];
      }
    }

    // 保存特征点总地图，比较稀疏
    if(!SaveIntensity)
    {
      *globalMapKeyFrames = *globalCornerMapKeyFrames + *globalSurfMapKeyFrames;
    }

    // downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    // downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);

    // 转成RGB点云,容易观察
    laserCloudFullResColor->clear();
    int laserCloudFullResNum = globalMapKeyFrames->points.size();
    for (int i = 0; i < laserCloudFullResNum; i++) {
      pcl::PointXYZRGB temp_point;
      RGBpointAssociateToMap(&globalMapKeyFrames->points[i], &temp_point);
      laserCloudFullResColor->push_back(temp_point);
    }

    // 降采样
    // 转成RGB点云,容易观察
    // laserCloudFullResColor->clear();
    // int laserCloudFullResNum = globalMapKeyFramesDS->points.size();
    // for (int i = 0; i < laserCloudFullResNum; i++) {
    //   pcl::PointXYZRGB temp_point;
    //   RGBpointAssociateToMap(&globalMapKeyFramesDS->points[i], &temp_point);
    //   laserCloudFullResColor->push_back(temp_point);
    // }

    sensor_msgs::PointCloud2 cloudMsgTemp;
    // pcl::toROSMsg(*globalMapKeyFramesDS, cloudMsgTemp);
    // pcl::toROSMsg(*laserCloudFullResColor, cloudMsgTemp);
    pcl::toROSMsg(*globalMapKeyFrames, cloudMsgTemp);
    cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
    cloudMsgTemp.header.frame_id = "camera_init";
    pubLaserCloudSurround.publish(cloudMsgTemp);

    globalMapKeyPoses->clear();
    globalMapKeyPosesDS->clear();
    globalMapKeyFrames->clear();
  }

  void loopClosureThread() {
    if (loopClosureEnableFlag == false) return;

    ros::Rate rate(1);
    while (ros::ok()) {
      rate.sleep();
      performLoopClosure();
      visualizeLoopClosure();
    }
  }

  bool detectLoopClosure_ori() {
    latestSurfKeyFrameCloud->clear();
    nearHistorySurfKeyFrameCloud->clear();
    nearHistorySurfKeyFrameCloudDS->clear();

    std::lock_guard<std::mutex> lock(mtx);

    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
    kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);
    kdtreeHistoryKeyPoses->radiusSearch(
        currentRobotPosPoint, historyKeyframeSearchRadius, pointSearchIndLoop,
        pointSearchSqDisLoop, 0);

    closestHistoryFrameID = -1;
    for (int i = 0; i < pointSearchIndLoop.size(); ++i) {
      int id = pointSearchIndLoop[i];
      if (abs(cloudKeyPoses6D->points[id].time - timeLaserOdometry) > 30.0) {
        closestHistoryFrameID = id;
        break;
      }
    }
    if (closestHistoryFrameID == -1) {
      return false;
    }

    latestFrameIDLoopCloure = cloudKeyPoses3D->points.size() - 1;
    *latestSurfKeyFrameCloud +=
        *transformPointCloud(cornerCloudKeyFrames[latestFrameIDLoopCloure],
                             &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
    *latestSurfKeyFrameCloud +=
        *transformPointCloud(surfCloudKeyFrames[latestFrameIDLoopCloure],
                             &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);

    pcl::PointCloud<PointType>::Ptr hahaCloud(new pcl::PointCloud<PointType>());
    int cloudSize = latestSurfKeyFrameCloud->points.size();
    for (int i = 0; i < cloudSize; ++i) {
      if ((int)latestSurfKeyFrameCloud->points[i].intensity >= 0) {
        hahaCloud->push_back(latestSurfKeyFrameCloud->points[i]);
      }
    }
    latestSurfKeyFrameCloud->clear();
    *latestSurfKeyFrameCloud = *hahaCloud;

    for (int j = -historyKeyframeSearchNum; j <= historyKeyframeSearchNum;
         ++j) {
      if (closestHistoryFrameID + j < 0 ||
          closestHistoryFrameID + j > latestFrameIDLoopCloure)
        continue;
      *nearHistorySurfKeyFrameCloud += *transformPointCloud(
          cornerCloudKeyFrames[closestHistoryFrameID + j],
          &cloudKeyPoses6D->points[closestHistoryFrameID + j]);
      *nearHistorySurfKeyFrameCloud += *transformPointCloud(
          surfCloudKeyFrames[closestHistoryFrameID + j],
          &cloudKeyPoses6D->points[closestHistoryFrameID + j]);
    }

    downSizeFilterHistoryKeyFrames.setInputCloud(nearHistorySurfKeyFrameCloud);
    downSizeFilterHistoryKeyFrames.filter(*nearHistorySurfKeyFrameCloudDS);

    if (pubHistoryKeyFrames.getNumSubscribers() != 0) {
      sensor_msgs::PointCloud2 cloudMsgTemp;
      pcl::toROSMsg(*nearHistorySurfKeyFrameCloudDS, cloudMsgTemp);
      cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
      cloudMsgTemp.header.frame_id = "camera_init";
      pubHistoryKeyFrames.publish(cloudMsgTemp);
    }

    return true;
  }

  bool detectLoopClosure(){

    std::lock_guard<std::mutex> lock(mtx);

    // cout<<"loop1"<<endl;
    /* 
      * 1. xyz distance-based radius search (contained in the original LeGO LOAM code)
      * - for fine-stichting trajectories (for not-recognized nodes within scan context search) 
      */
    // 先进行LEGO-LOAM中原始的基于里程计的回环检测,该方法精度高,但是召回率低
    RSlatestSurfKeyFrameCloud->clear();
    RSnearHistorySurfKeyFrameCloud->clear();
    RSnearHistorySurfKeyFrameCloudDS->clear();

    // find the closest history key frame
    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
    kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);
    kdtreeHistoryKeyPoses->radiusSearch(currentRobotPosPoint, historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
    
    RSclosestHistoryFrameID = -1;
    int curMinID = 1000000;
    // policy: take Oldest one (to fix error of the whole trajectory)
    for (int i = 0; i < pointSearchIndLoop.size(); ++i){
        int id = pointSearchIndLoop[i];
        // 时间间隔超过30秒
        if (abs(cloudKeyPoses6D->points[id].time - timeLaserOdometry) > 30.0){
            // RSclosestHistoryFrameID = id;
            // break;
            // 查找离当前帧时间最远的一帧
            if( id < curMinID ) {
                curMinID = id;
                RSclosestHistoryFrameID = curMinID;
            }
        }
    }

    // cout<<"RSclosestHistoryFrameID: "<<RSclosestHistoryFrameID<<endl;

    // 检测不到RS, 那我们开始检测SC
    if (RSclosestHistoryFrameID == -1){
        // Do nothing here
        // then, do the next check: Scan context-based search 
        // not return false here;
        // return false;
    }
    else {
        // save latest key frames
        // 保存最新一帧的点云(全)
        latestFrameIDLoopCloure = cloudKeyPoses3D->points.size() - 1;
        *RSlatestSurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[latestFrameIDLoopCloure], &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
        *RSlatestSurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[latestFrameIDLoopCloure],   &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
        pcl::PointCloud<PointType>::Ptr RShahaCloud(new pcl::PointCloud<PointType>());
        int cloudSize = RSlatestSurfKeyFrameCloud->points.size();
        for (int i = 0; i < cloudSize; ++i){
          // 过滤无效点
            if ((int)RSlatestSurfKeyFrameCloud->points[i].intensity >= 0){
                RShahaCloud->push_back(RSlatestSurfKeyFrameCloud->points[i]);
            }
        }
        RSlatestSurfKeyFrameCloud->clear();
        *RSlatestSurfKeyFrameCloud = *RShahaCloud;

        // save history near key frames
        // 闭环候选帧前后各25帧的点云组合形成大点云,用于与当前帧进行ICP
        for (int j = -historyKeyframeSearchNum; j <= historyKeyframeSearchNum; ++j){
            if (RSclosestHistoryFrameID + j < 0 || RSclosestHistoryFrameID + j > latestFrameIDLoopCloure)
                continue;
            *RSnearHistorySurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[RSclosestHistoryFrameID+j], &cloudKeyPoses6D->points[RSclosestHistoryFrameID+j]);
            *RSnearHistorySurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[RSclosestHistoryFrameID+j],   &cloudKeyPoses6D->points[RSclosestHistoryFrameID+j]);
        }
        downSizeFilterHistoryKeyFrames.setInputCloud(RSnearHistorySurfKeyFrameCloud);
        downSizeFilterHistoryKeyFrames.filter(*RSnearHistorySurfKeyFrameCloudDS);
    }

    /* 
      * 2. Scan context-based global localization 
      */
    SClatestSurfKeyFrameCloud->clear();
    SCnearHistorySurfKeyFrameCloud->clear();
    SCnearHistorySurfKeyFrameCloudDS->clear();

    // std::lock_guard<std::mutex> lock(mtx);        
    latestFrameIDLoopCloure = cloudKeyPoses3D->points.size() - 1;
    SCclosestHistoryFrameID = -1; // init with -1
    // 若匹配成功,返回闭环帧索引和方位角偏差
    auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff 
    // 闭环候选帧索引
    SCclosestHistoryFrameID = detectResult.first;
    // 方位角偏差
    yawDiffRad = detectResult.second; // not use for v1 (because pcl icp withi initial somthing wrong...)

    // if all close, reject
    // 检测不到闭环
    // if (SCclosestHistoryFrameID == -1){ 
    //   if(RSclosestHistoryFrameID != -1)
    //   {
    //     return true;
    //   }
    //     return false;
    // }
    if (SCclosestHistoryFrameID == -1 && RSclosestHistoryFrameID == -1){ 
        return false;
    }
    else if(SCclosestHistoryFrameID == -1 && RSclosestHistoryFrameID != -1)
    {
      return true;
    }

    // save latest key frames: query ptcloud (corner points + surface points)
    // NOTE: using "closestHistoryFrameID" to make same root of submap points to get a direct relative between the query point cloud (latestSurfKeyFrameCloud) and the map (nearHistorySurfKeyFrameCloud). by giseop
    // i.e., set the query point cloud within mapside's local coordinate
    *SClatestSurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[latestFrameIDLoopCloure], &cloudKeyPoses6D->points[SCclosestHistoryFrameID]);         
    *SClatestSurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[latestFrameIDLoopCloure],   &cloudKeyPoses6D->points[SCclosestHistoryFrameID]); 

    // 获取闭环帧的完整点云
    pcl::PointCloud<PointType>::Ptr SChahaCloud(new pcl::PointCloud<PointType>());
    int cloudSize = SClatestSurfKeyFrameCloud->points.size();
    for (int i = 0; i < cloudSize; ++i){
        if ((int)SClatestSurfKeyFrameCloud->points[i].intensity >= 0){
            SChahaCloud->push_back(SClatestSurfKeyFrameCloud->points[i]);
        }
    }
    SClatestSurfKeyFrameCloud->clear();
    *SClatestSurfKeyFrameCloud = *SChahaCloud;

  // save history near key frames: map ptcloud (icp to query ptcloud)
//    处理匹配闭环帧前后historyKeyframeSearchNum(25)帧的关键帧,形成点云用于进行icp
    for (int j = -historyKeyframeSearchNum; j <= historyKeyframeSearchNum; ++j){
        if (SCclosestHistoryFrameID + j < 0 || SCclosestHistoryFrameID + j > latestFrameIDLoopCloure)
            continue;
        *SCnearHistorySurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[SCclosestHistoryFrameID+j], &cloudKeyPoses6D->points[SCclosestHistoryFrameID+j]);
        *SCnearHistorySurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[SCclosestHistoryFrameID+j],   &cloudKeyPoses6D->points[SCclosestHistoryFrameID+j]);
    }
    downSizeFilterHistoryKeyFrames.setInputCloud(SCnearHistorySurfKeyFrameCloud);
    downSizeFilterHistoryKeyFrames.filter(*SCnearHistorySurfKeyFrameCloudDS);

    return true;
  } // detectLoopClosure

  void performLoopClosure() {
    if (cloudKeyPoses3D->points.empty() == true) return;

    if (potentialLoopFlag == false) {
      if (detectLoopClosure() == true) {
        potentialLoopFlag = true;
        timeSaveFirstCurrentScanForLoopClosure = timeLaserOdometry;
      }
      if (potentialLoopFlag == false) return;
    }

    potentialLoopFlag = false;

    // *****
    // Main 
    // *****
    // make common variables at forward
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionCameraFrame;
    float noiseScore = 0.5; // constant is ok...
    gtsam::Vector Vector6(6);
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    constraintNoise = noiseModel::Diagonal::Variances(Vector6);
    robustNoiseModel = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure
        gtsam::noiseModel::Diagonal::Variances(Vector6)
    ); // - checked it works. but with robust kernel, map modification may be delayed (i.e,. requires more true-positive loop factors)

    
    bool isValidRSloopFactor = false;
    bool isValidSCloopFactor = false;

    /*
        * 1. RS loop factor (radius search)
        */
    // 检查LEGO-LOAM原来的回环检测是否成功
    if( RSclosestHistoryFrameID != -1 ) {
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(100);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        icp.setInputSource(RSlatestSurfKeyFrameCloud);
        icp.setInputTarget(RSnearHistorySurfKeyFrameCloudDS);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        std::cout << "[RS] ICP fit score: " << icp.getFitnessScore() << std::endl;
        if ( icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore ) {
            std::cout << "[RS] Reject this loop (bad icp fit score, > " << historyKeyframeFitnessScore << ")" << std::endl;
            isValidRSloopFactor = false;
        }
        else {
            std::cout << "[RS] The detected loop factor is added between Current [ " << latestFrameIDLoopCloure << " ] and RS nearest [ " << RSclosestHistoryFrameID << " ]" << std::endl;
            isValidRSloopFactor = true;
        }

        if( isValidRSloopFactor == true ) {
            correctionCameraFrame = icp.getFinalTransformation(); // get transformation in camera frame (because points are in camera frame)
            pcl::getTranslationAndEulerAngles(correctionCameraFrame, x, y, z, roll, pitch, yaw);
            Eigen::Affine3f correctionLidarFrame = pcl::getTransformation(z, x, y, yaw, roll, pitch);
            // transform from world origin to wrong pose
            Eigen::Affine3f tWrong = pclPointToAffine3fCameraToLidar(cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
            // transform from world origin to corrected pose
            Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong; // pre-multiplying -> successive rotation about a fixed frame
            pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
            gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
            gtsam::Pose3 poseTo = pclPointTogtsamPose3(cloudKeyPoses6D->points[RSclosestHistoryFrameID]);
            gtsam::Vector Vector6(6);

            std::lock_guard<std::mutex> lock(mtx);
            gtSAMgraph.add(BetweenFactor<Pose3>(X(latestFrameIDLoopCloure), X(RSclosestHistoryFrameID), poseFrom.between(poseTo), robustNoiseModel));
            isam->update(gtSAMgraph);
            isam->update();
            gtSAMgraph.resize(0);
            // #kyle
            // flagging
            loopIndexContainer[latestFrameIDLoopCloure] = RSclosestHistoryFrameID;
            // aLoopIsClosed = true;
            // return;
        }
    }

    /*
        * 2. SC loop factor (scan context)
        */
    if( SCclosestHistoryFrameID != -1 ) {
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(100);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        // Eigen::Affine3f icpInitialMatFoo = pcl::getTransformation(0, 0, 0, yawDiffRad, 0, 0); // because within cam coord: (z, x, y, yaw, roll, pitch)
        // Eigen::Matrix4f icpInitialMat = icpInitialMatFoo.matrix();
        icp.setInputSource(SClatestSurfKeyFrameCloud);
        icp.setInputTarget(SCnearHistorySurfKeyFrameCloudDS);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result); 
        // icp.align(*unused_result, icpInitialMat); // PCL icp non-eye initial is bad ... don't use (LeGO LOAM author also said pcl transform is weird.)

        std::cout << "[SC] ICP fit score: " << icp.getFitnessScore() << std::endl;
        if ( icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore ) {
            std::cout << "[SC] Reject this loop (bad icp fit score, > " << historyKeyframeFitnessScore << ")" << std::endl;
            isValidSCloopFactor = false;
        }
        else {
            std::cout << "[SC] The detected loop factor is added between Current [ " << latestFrameIDLoopCloure << " ] and SC nearest [ " << SCclosestHistoryFrameID << " ]" << std::endl;
            isValidSCloopFactor = true;
        }

        if( isValidSCloopFactor == true ) {
            correctionCameraFrame = icp.getFinalTransformation(); // get transformation in camera frame (because points are in camera frame)
            pcl::getTranslationAndEulerAngles (correctionCameraFrame, x, y, z, roll, pitch, yaw);
            gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
            gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

            std::lock_guard<std::mutex> lock(mtx);
            // gtSAMgraph.add(BetweenFactor<Pose3>(latestFrameIDLoopCloure, closestHistoryFrameID, poseFrom.between(poseTo), constraintNoise)); // original 
            gtSAMgraph.add(BetweenFactor<Pose3>(X(latestFrameIDLoopCloure), X(SCclosestHistoryFrameID), poseFrom.between(poseTo), robustNoiseModel)); // giseop
            isam->update(gtSAMgraph);
            isam->update();
            gtSAMgraph.resize(0);
            // #kyle
            // flagging
            // aLoopIsClosed = true;
            // return;
        }
    }
    // flagging
    aLoopIsClosed = true;
  } //performLoopClosure

  Pose3 pclPointTogtsamPose3(PointTypePose thisPoint) {
    return Pose3(
        Rot3::RzRyRx(double(thisPoint.yaw), double(thisPoint.roll),
                     double(thisPoint.pitch)),
        Point3(double(thisPoint.z), double(thisPoint.x), double(thisPoint.y)));
  }

  Eigen::Affine3f pclPointToAffine3fCameraToLidar(PointTypePose thisPoint) {
    return pcl::getTransformation(thisPoint.z, thisPoint.x, thisPoint.y,
                                  thisPoint.yaw, thisPoint.roll,
                                  thisPoint.pitch);
  }

  // 提取附近的关键帧
  void extractSurroundingKeyFrames() {
    if (cloudKeyPoses3D->points.empty() == true) return;

    // 是否触发闭环?
    if (loopClosureEnableFlag == true) {
      if (recentCornerCloudKeyFrames.size() < surroundingKeyframeSearchNum) {
        recentCornerCloudKeyFrames.clear();
        recentSurfCloudKeyFrames.clear();
        recentOutlierCloudKeyFrames.clear();
        int numPoses = cloudKeyPoses3D->points.size();
        for (int i = numPoses - 1; i >= 0; --i) {
          int thisKeyInd = (int)cloudKeyPoses3D->points[i].intensity;
          PointTypePose thisTransformation =
              cloudKeyPoses6D->points[thisKeyInd];
          updateTransformPointCloudSinCos(&thisTransformation);
          recentCornerCloudKeyFrames.push_front(
              transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));
          recentSurfCloudKeyFrames.push_front(
              transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
          recentOutlierCloudKeyFrames.push_front(
              transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
          if (recentCornerCloudKeyFrames.size() >= surroundingKeyframeSearchNum)
            break;
        }
      } else {
        if (latestFrameID != cloudKeyPoses3D->points.size() - 1) {
          recentCornerCloudKeyFrames.pop_front();
          recentSurfCloudKeyFrames.pop_front();
          recentOutlierCloudKeyFrames.pop_front();
          latestFrameID = cloudKeyPoses3D->points.size() - 1;
          PointTypePose thisTransformation =
              cloudKeyPoses6D->points[latestFrameID];
          updateTransformPointCloudSinCos(&thisTransformation);
          recentCornerCloudKeyFrames.push_back(
              transformPointCloud(cornerCloudKeyFrames[latestFrameID]));
          recentSurfCloudKeyFrames.push_back(
              transformPointCloud(surfCloudKeyFrames[latestFrameID]));
          recentOutlierCloudKeyFrames.push_back(
              transformPointCloud(outlierCloudKeyFrames[latestFrameID]));
        }
      }

      for (int i = 0; i < recentCornerCloudKeyFrames.size(); ++i) {
        *laserCloudCornerFromMap += *recentCornerCloudKeyFrames[i];
        *laserCloudSurfFromMap += *recentSurfCloudKeyFrames[i];
        *laserCloudSurfFromMap += *recentOutlierCloudKeyFrames[i];
      }
    } else {
      surroundingKeyPoses->clear();
      surroundingKeyPosesDS->clear();

      kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);
      kdtreeSurroundingKeyPoses->radiusSearch(
          currentRobotPosPoint, (double)surroundingKeyframeSearchRadius,
          pointSearchInd, pointSearchSqDis, 0);
      for (int i = 0; i < pointSearchInd.size(); ++i)
        surroundingKeyPoses->points.push_back(
            cloudKeyPoses3D->points[pointSearchInd[i]]);
      downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
      downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

      int numSurroundingPosesDS = surroundingKeyPosesDS->points.size();
      for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i) {
        bool existingFlag = false;
        for (int j = 0; j < numSurroundingPosesDS; ++j) {
          if (surroundingExistingKeyPosesID[i] ==
              (int)surroundingKeyPosesDS->points[j].intensity) {
            existingFlag = true;
            break;
          }
        }
        if (existingFlag == false) {
          surroundingExistingKeyPosesID.erase(
              surroundingExistingKeyPosesID.begin() + i);
          surroundingCornerCloudKeyFrames.erase(
              surroundingCornerCloudKeyFrames.begin() + i);
          surroundingSurfCloudKeyFrames.erase(
              surroundingSurfCloudKeyFrames.begin() + i);
          surroundingOutlierCloudKeyFrames.erase(
              surroundingOutlierCloudKeyFrames.begin() + i);
          --i;
        }
      }

      for (int i = 0; i < numSurroundingPosesDS; ++i) {
        bool existingFlag = false;
        for (auto iter = surroundingExistingKeyPosesID.begin();
             iter != surroundingExistingKeyPosesID.end(); ++iter) {
          if ((*iter) == (int)surroundingKeyPosesDS->points[i].intensity) {
            existingFlag = true;
            break;
          }
        }
        if (existingFlag == true) {
          continue;
        } else {
          int thisKeyInd = (int)surroundingKeyPosesDS->points[i].intensity;
          PointTypePose thisTransformation =
              cloudKeyPoses6D->points[thisKeyInd];
          updateTransformPointCloudSinCos(&thisTransformation);
          surroundingExistingKeyPosesID.push_back(thisKeyInd);
          surroundingCornerCloudKeyFrames.push_back(
              transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));
          surroundingSurfCloudKeyFrames.push_back(
              transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
          surroundingOutlierCloudKeyFrames.push_back(
              transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
        }
      }

      for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i) {
        *laserCloudCornerFromMap += *surroundingCornerCloudKeyFrames[i];
        *laserCloudSurfFromMap += *surroundingSurfCloudKeyFrames[i];
        *laserCloudSurfFromMap += *surroundingOutlierCloudKeyFrames[i];
      }
    }

    downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
    downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
    laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->points.size();

    downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
    downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
    laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->points.size();
  }

  void downsampleCurrentScan() {
    // scan context
    laserCloudRawDS->clear();
    // pcl::PointCloud<PointType>::Ptr laserCloudRawDS_temp(
    //     new pcl::PointCloud<PointType>());

    downSizeFilterScancontext.setInputCloud(laserCloudRaw);
    downSizeFilterScancontext.filter(*laserCloudRawDS);

    // 转置点云
    // size_t cloudSize = laserCloudRawDS->points.size();
    // PointType thisPoint;
    // for(size_t i = 0; i < cloudSize; ++i)
    // {
    //   // copy该点用于操作
    //   thisPoint.x = laserCloudRawDS->points[i].x;
    //   thisPoint.y = laserCloudRawDS->points[i].y;
    //   thisPoint.z = laserCloudRawDS->points[i].z;

    //   laserCloudRawDS->points[i].x = thisPoint.y;
    //   laserCloudRawDS->points[i].y = thisPoint.z;
    //   laserCloudRawDS->points[i].z = thisPoint.x;
    // }
    
    laserCloudCornerLastDS->clear();
    downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
    downSizeFilterCorner.filter(*laserCloudCornerLastDS);
    laserCloudCornerLastDSNum = laserCloudCornerLastDS->points.size();

    laserCloudSurfLastDS->clear();
    downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
    downSizeFilterSurf.filter(*laserCloudSurfLastDS);
    laserCloudSurfLastDSNum = laserCloudSurfLastDS->points.size();

    laserCloudOutlierLastDS->clear();
    downSizeFilterOutlier.setInputCloud(laserCloudOutlierLast);
    downSizeFilterOutlier.filter(*laserCloudOutlierLastDS);
    laserCloudOutlierLastDSNum = laserCloudOutlierLastDS->points.size();

    laserCloudSurfTotalLast->clear();
    laserCloudSurfTotalLastDS->clear();
    *laserCloudSurfTotalLast += *laserCloudSurfLastDS;
    *laserCloudSurfTotalLast += *laserCloudOutlierLastDS;
    downSizeFilterSurf.setInputCloud(laserCloudSurfTotalLast);
    downSizeFilterSurf.filter(*laserCloudSurfTotalLastDS);
    laserCloudSurfTotalLastDSNum = laserCloudSurfTotalLastDS->points.size();
  }

  void cornerOptimization(int iterCount) {
    updatePointAssociateToMapSinCos();
    for (int i = 0; i < laserCloudCornerLastDSNum; i++) {
      pointOri = laserCloudCornerLastDS->points[i];
      pointAssociateToMap(&pointOri, &pointSel);
      kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd,
                                          pointSearchSqDis);

      if (pointSearchSqDis[4] < 1.0) {
        float cx = 0, cy = 0, cz = 0;
        for (int j = 0; j < 5; j++) {
          cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
          cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
          cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
        }
        cx /= 5;
        cy /= 5;
        cz /= 5;

        float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
        for (int j = 0; j < 5; j++) {
          float ax =
              laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
          float ay =
              laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
          float az =
              laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

          a11 += ax * ax;
          a12 += ax * ay;
          a13 += ax * az;
          a22 += ay * ay;
          a23 += ay * az;
          a33 += az * az;
        }
        a11 /= 5;
        a12 /= 5;
        a13 /= 5;
        a22 /= 5;
        a23 /= 5;
        a33 /= 5;

        matA1.at<float>(0, 0) = a11;
        matA1.at<float>(0, 1) = a12;
        matA1.at<float>(0, 2) = a13;
        matA1.at<float>(1, 0) = a12;
        matA1.at<float>(1, 1) = a22;
        matA1.at<float>(1, 2) = a23;
        matA1.at<float>(2, 0) = a13;
        matA1.at<float>(2, 1) = a23;
        matA1.at<float>(2, 2) = a33;

        cv::eigen(matA1, matD1, matV1);

        if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {
          float x0 = pointSel.x;
          float y0 = pointSel.y;
          float z0 = pointSel.z;
          float x1 = cx + 0.1 * matV1.at<float>(0, 0);
          float y1 = cy + 0.1 * matV1.at<float>(0, 1);
          float z1 = cz + 0.1 * matV1.at<float>(0, 2);
          float x2 = cx - 0.1 * matV1.at<float>(0, 0);
          float y2 = cy - 0.1 * matV1.at<float>(0, 1);
          float z2 = cz - 0.1 * matV1.at<float>(0, 2);

          float a012 =
              sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) *
                       ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
                   ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) *
                       ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                   ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) *
                       ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

          float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) +
                           (z1 - z2) * (z1 - z2));

          float la =
              ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
               (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) /
              a012 / l12;

          float lb =
              -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) -
                (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
              a012 / l12;

          float lc =
              -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
              a012 / l12;

          float ld2 = a012 / l12;

          float s = 1 - 0.9 * fabs(ld2);

          coeff.x = s * la;
          coeff.y = s * lb;
          coeff.z = s * lc;
          coeff.intensity = s * ld2;

          if (s > 0.1) {
            laserCloudOri->push_back(pointOri);
            coeffSel->push_back(coeff);
          }
        }
      }
    }
  }

  void surfOptimization(int iterCount) {
    updatePointAssociateToMapSinCos();
    for (int i = 0; i < laserCloudSurfTotalLastDSNum; i++) {
      pointOri = laserCloudSurfTotalLastDS->points[i];
      pointAssociateToMap(&pointOri, &pointSel);
      kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd,
                                        pointSearchSqDis);

      if (pointSearchSqDis[4] < 1.0) {
        for (int j = 0; j < 5; j++) {
          matA0.at<float>(j, 0) =
              laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
          matA0.at<float>(j, 1) =
              laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
          matA0.at<float>(j, 2) =
              laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
        }
        cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);

        float pa = matX0.at<float>(0, 0);
        float pb = matX0.at<float>(1, 0);
        float pc = matX0.at<float>(2, 0);
        float pd = 1;

        float ps = sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        bool planeValid = true;
        for (int j = 0; j < 5; j++) {
          if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                   pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                   pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z +
                   pd) > 0.2) {
            planeValid = false;
            break;
          }
        }

        if (planeValid) {
          float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

          float s = 1 - 0.9 * fabs(pd2) /
                            sqrt(sqrt(pointSel.x * pointSel.x +
                                      pointSel.y * pointSel.y +
                                      pointSel.z * pointSel.z));

          coeff.x = s * pa;
          coeff.y = s * pb;
          coeff.z = s * pc;
          coeff.intensity = s * pd2;

          if (s > 0.1) {
            laserCloudOri->push_back(pointOri);
            coeffSel->push_back(coeff);
          }
        }
      }
    }
  }

  bool LMOptimization(int iterCount) {
    float srx = sin(transformTobeMapped[0]);
    float crx = cos(transformTobeMapped[0]);
    float sry = sin(transformTobeMapped[1]);
    float cry = cos(transformTobeMapped[1]);
    float srz = sin(transformTobeMapped[2]);
    float crz = cos(transformTobeMapped[2]);

    int laserCloudSelNum = laserCloudOri->points.size();
    if (laserCloudSelNum < 50) {
      return false;
    }

    cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
    for (int i = 0; i < laserCloudSelNum; i++) {
      pointOri = laserCloudOri->points[i];
      coeff = coeffSel->points[i];

      float arx = (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y -
                   srx * sry * pointOri.z) *
                      coeff.x +
                  (-srx * srz * pointOri.x - crz * srx * pointOri.y -
                   crx * pointOri.z) *
                      coeff.y +
                  (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y -
                   cry * srx * pointOri.z) *
                      coeff.z;

      float ary = ((cry * srx * srz - crz * sry) * pointOri.x +
                   (sry * srz + cry * crz * srx) * pointOri.y +
                   crx * cry * pointOri.z) *
                      coeff.x +
                  ((-cry * crz - srx * sry * srz) * pointOri.x +
                   (cry * srz - crz * srx * sry) * pointOri.y -
                   crx * sry * pointOri.z) *
                      coeff.z;

      float arz = ((crz * srx * sry - cry * srz) * pointOri.x +
                   (-cry * crz - srx * sry * srz) * pointOri.y) *
                      coeff.x +
                  (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y +
                  ((sry * srz + cry * crz * srx) * pointOri.x +
                   (crz * sry - cry * srx * srz) * pointOri.y) *
                      coeff.z;

      matA.at<float>(i, 0) = arx;
      matA.at<float>(i, 1) = ary;
      matA.at<float>(i, 2) = arz;
      matA.at<float>(i, 3) = coeff.x;
      matA.at<float>(i, 4) = coeff.y;
      matA.at<float>(i, 5) = coeff.z;
      matB.at<float>(i, 0) = -coeff.intensity;
    }
    cv::transpose(matA, matAt);
    matAtA = matAt * matA;
    matAtB = matAt * matB;
    cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

    if (iterCount == 0) {
      cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
      cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
      cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

      cv::eigen(matAtA, matE, matV);
      matV.copyTo(matV2);

      isDegenerate = false;
      float eignThre[6] = {100, 100, 100, 100, 100, 100};
      for (int i = 5; i >= 0; i--) {
        if (matE.at<float>(0, i) < eignThre[i]) {
          for (int j = 0; j < 6; j++) {
            matV2.at<float>(i, j) = 0;
          }
          isDegenerate = true;
        } else {
          break;
        }
      }
      matP = matV.inv() * matV2;
    }

    if (isDegenerate) {
      cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
      matX.copyTo(matX2);
      matX = matP * matX2;
    }

    transformTobeMapped[0] += matX.at<float>(0, 0);
    transformTobeMapped[1] += matX.at<float>(1, 0);
    transformTobeMapped[2] += matX.at<float>(2, 0);
    transformTobeMapped[3] += matX.at<float>(3, 0);
    transformTobeMapped[4] += matX.at<float>(4, 0);
    transformTobeMapped[5] += matX.at<float>(5, 0);

    float deltaR = sqrt(pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                        pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                        pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
    float deltaT = sqrt(pow(matX.at<float>(3, 0) * 100, 2) +
                        pow(matX.at<float>(4, 0) * 100, 2) +
                        pow(matX.at<float>(5, 0) * 100, 2));

    if (deltaR < 0.05 && deltaT < 0.05) {
      return true;
    }
    return false;
  }

  void scan2MapOptimization() {
    if (laserCloudCornerFromMapDSNum > 10 && laserCloudSurfFromMapDSNum > 100) {
      kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
      kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

      for (int iterCount = 0; iterCount < 10; iterCount++) {
        laserCloudOri->clear();
        coeffSel->clear();

        cornerOptimization(iterCount);
        surfOptimization(iterCount);

        if (LMOptimization(iterCount) == true) break;
      }

      transformUpdate();
    }
  }

  Eigen::Vector3d OdomUTMPos;
  Eigen::Vector3d LastOdomUTMPos;
  Eigen::Vector3d SumRandom;

  void saveKeyFramesAndFactor() {
    currentRobotPosPoint.x = transformAftMapped[3];
    currentRobotPosPoint.y = transformAftMapped[4];
    currentRobotPosPoint.z = transformAftMapped[5];

    bool saveThisKeyFrame = true;
    if (sqrt((previousRobotPosPoint.x - currentRobotPosPoint.x) *
                 (previousRobotPosPoint.x - currentRobotPosPoint.x) +
             (previousRobotPosPoint.y - currentRobotPosPoint.y) *
                 (previousRobotPosPoint.y - currentRobotPosPoint.y) +
             (previousRobotPosPoint.z - currentRobotPosPoint.z) *
                 (previousRobotPosPoint.z - currentRobotPosPoint.z)) < 0.3) {
      saveThisKeyFrame = false;
    }

    if (saveThisKeyFrame == false && !cloudKeyPoses3D->points.empty()) return;

    previousRobotPosPoint = currentRobotPosPoint;

    if (cloudKeyPoses3D->points.empty()) {
      gtSAMgraph.add(PriorFactor<Pose3>(
          X(cloudKeyPoses3D->points.size()),
          Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0],
                             transformTobeMapped[1]),
                Point3(transformTobeMapped[5], transformTobeMapped[3],
                       transformTobeMapped[4])),
          priorNoise));
      initialEstimate.insert(
          X(cloudKeyPoses3D->points.size()), Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0],
                                transformTobeMapped[1]),
                   Point3(transformTobeMapped[5], transformTobeMapped[3],
                          transformTobeMapped[4])));
      for (int i = 0; i < 6; ++i) transformLast[i] = transformTobeMapped[i];

      // // find the beginning imu
      // double InitImuTime = -1.0;
      // while(!imuQue.empty())
      // {
      //     InitImuTime = imuQue.front().header.stamp.toSec();
      //     if(InitImuTime < timeLaserOdometry - 0.02)
      //     {
      //         imuQue.pop_front();
      //     }
      //     else
      //         break;
      // }
      // lastImuTime = InitImuTime;
      // imuQue.pop_front();

      // initialEstimate.insert(V(cloudKeyPoses3D->points.size()), prior_velocity);
      // initialEstimate.insert(B(cloudKeyPoses3D->points.size()), prior_imu_bias);

      // gtSAMgraph.add(PriorFactor<Vector3>(V(cloudKeyPoses3D->points.size()), prior_velocity, velocity_noise_model));
      // gtSAMgraph.add(PriorFactor<imuBias::ConstantBias>(B(cloudKeyPoses3D->points.size()), prior_imu_bias, bias_noise_model));

    } 
    // else {
    //   gtsam::Pose3 poseFrom = Pose3(
    //       Rot3::RzRyRx(transformLast[2], transformLast[0], transformLast[1]),
    //       Point3(transformLast[5], transformLast[3], transformLast[4]));
    //   gtsam::Pose3 poseTo =
    //       Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0],
    //                          transformAftMapped[1]),
    //             Point3(transformAftMapped[5], transformAftMapped[3],
    //                    transformAftMapped[4]));
    //   initialEstimate.insert(
    //       X(cloudKeyPoses3D->points.size()),
    //       Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0],
    //                          transformAftMapped[1]),
    //             Point3(transformAftMapped[5], transformAftMapped[3],
    //                    transformAftMapped[4])));
    //   gtSAMgraph.add(BetweenFactor<Pose3>(
    //       X(cloudKeyPoses3D->points.size()-1), X(cloudKeyPoses3D->points.size()),
    //       poseFrom.between(poseTo), odometryNoise));

    //   // // performing preintegration
    //   // sensor_msgs::Imu thisImu;
    //   // while(!imuQue.empty())
    //   // {
    //   //     if(imuQue.front().header.stamp.toSec()<timeLaserOdometry)
    //   //     {
    //   //         // extract the early imu
    //   //         thisImu = imuQue.front();
    //   //         imuQue.pop_front();
    //   //         curImuTime = thisImu.header.stamp.toSec();
    //   //         double Imu_dt = curImuTime - lastImuTime;
    //   //         // cout<<"dt:"<<Imu_dt<<endl;
    //   //         lastImuTime = curImuTime;
    //   //         // integrate this single imu message
    //   //         imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
    //   //                                                 gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), Imu_dt);
    //   //     }
    //   //     else
    //   //         break;
    //   // }

    //   // // generate imu factor
    //   // const PreintegratedImuMeasurements& preint_imu =
    //   //     dynamic_cast<const PreintegratedImuMeasurements&>(
    //   //       *imuIntegratorImu_);
    //   // ImuFactor imu_factor(X(cloudKeyPoses3D->points.size()-1), V(cloudKeyPoses3D->points.size()-1),
    //   //                       X(cloudKeyPoses3D->points.size()  ), V(cloudKeyPoses3D->points.size()  ),
    //   //                       B(cloudKeyPoses3D->points.size()-1),
    //   //                       preint_imu);
    //   // gtSAMgraph.add(imu_factor);
    //   // imuBias::ConstantBias zero_bias(Vector3(0, 0, 0), Vector3(0, 0, 0));
    //   // gtSAMgraph.add(BetweenFactor<imuBias::ConstantBias>(B(cloudKeyPoses3D->points.size()-1), B(cloudKeyPoses3D->points.size()),
    //   //                                     zero_bias, bias_noise_model));

    //   // // Now optimize and compare results.
    //   // prop_state = imuIntegratorImu_->predict(prev_state, prev_bias);
    //   // prev_state = prop_state;
    //   // // initialEstimate.insert(X(cloudKeyPoses3D->points.size()), prop_state.pose());
    //   // initialEstimate.insert(V(cloudKeyPoses3D->points.size()), prop_state.v());
    //   // initialEstimate.insert(B(cloudKeyPoses3D->points.size()), prev_bias);

    //   // // cout<<"IMU position: "<<prop_state.v().transpose()<<endl;
    // }
        else if(useBetweenFactor)
        {
            gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(transformLast[2], transformLast[0], transformLast[1]),
                                                Point3(transformLast[5], transformLast[3], transformLast[4]));
            gtsam::Pose3 poseTo   = Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]),
                                                Point3(transformAftMapped[5], transformAftMapped[3], transformAftMapped[4]));
			
            // 构造函数原型:BetweenFactor (Key key1, Key key2, const VALUE &measured, const SharedNoiseModel &model)
            // 添加因子
            // gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->points.size()-1, cloudKeyPoses3D->points.size(), poseFrom.between(poseTo), odometryNoise));
            // initialEstimate.insert(cloudKeyPoses3D->points.size(), Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]),
            //                                                          		   Point3(transformAftMapped[5], transformAftMapped[3], transformAftMapped[4])));
            initialEstimate.insert(
                X(cloudKeyPoses3D->points.size()),
                Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0],
                                    transformAftMapped[1]),
                    Point3(transformAftMapped[5], transformAftMapped[3],
                            transformAftMapped[4])));
            gtSAMgraph.add(BetweenFactor<Pose3>(
                X(cloudKeyPoses3D->points.size()-1), X(cloudKeyPoses3D->points.size()),
                poseFrom.between(poseTo), odometryNoise));
        }
        else if(!useBetweenFactor) // for test
        {
            // gtSAMgraph.add(PriorFactor<Pose3>(cloudKeyPoses3D->points.size(), Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
            //                                            		 Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])), LidarNoise));
            // // initialEstimate的数据类型是Values,其实就是一个map，这里在0对应的值下面保存了一个Pose3
            // // 也就是插入该顶点的初值用于迭代
            // initialEstimate.insert(cloudKeyPoses3D->points.size(), Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
            //                                       Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])));

            gtSAMgraph.add(PriorFactor<Pose3>(X(cloudKeyPoses3D->points.size()), Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
                                                       		 Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])), LidarNoise));
            // initialEstimate的数据类型是Values,其实就是一个map，这里在0对应的值下面保存了一个Pose3
            // 也就是插入该顶点的初值用于迭代
            initialEstimate.insert(X(cloudKeyPoses3D->points.size()), Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
                                                  Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])));
        }

    // gps odometry for sleipnir
    if(!sleipnir_gps_queue.empty() && customized_gps_msg)
    {
      sleipnir_msgs::sensorgps lastGps;
      sleipnir_msgs::sensorgps currGps;
      sleipnir_msgs::sensorgps initGps;
      // extract the current gps message
      initGps = sleipnir_gps_queue.front();
      // cout<<"sleipnir_gps_queue.size(): "<<sleipnir_gps_queue.size()<<endl;
      while(!sleipnir_gps_queue.empty())
      {
        if(sleipnir_gps_queue.front().header.stamp.toSec()<timeLaserOdometry - 0.1)
        {
          lastGps = sleipnir_gps_queue.front();
          sleipnir_gps_queue.pop();
        }
        else
          break;
      }
      // cout<<"sleipnir_gps_queue.size(): "<<sleipnir_gps_queue.size()<<endl;
      if(!sleipnir_gps_queue.empty()) // there are proper data in the queue
      {
        // cout<<"debug: "<<sleipnir_gps_queue.front().header.stamp.toSec()-timeLaserOdometry<<endl;
        if(sleipnir_gps_queue.front().header.stamp.toSec()>timeLaserOdometry-0.1 
        && sleipnir_gps_queue.front().header.stamp.toSec()-timeLaserOdometry<0.1)
        {
          currGps = sleipnir_gps_queue.front();
          sleipnir_msgs::sensorgps newGps = GpsSlerp(lastGps, currGps, timeLaserOdometry);
          // cout<<"timeLaserOdometry: "<<timeLaserOdometry<<endl;
          // sleipnir_msgs::sensorgps newGps = currGps;
          // cout<<newGps<<endl<<endl;
          if(!init_fix)
          {
              newGps = currGps;
              // cout<<setprecision(15)<<"newGps.lat<<"<<newGps.lat<<endl;
              // cout<<setprecision(15)<<"newGps.lon<<"<<newGps.lon<<endl;
              // cout<<setprecision(15)<<"newGps.alt<<"<<newGps.alt<<endl;
              // cout<<setprecision(15)<<"newGps.heading<<"<<newGps.heading<<endl;
              // cout<<setprecision(15)<<"newGps.pitch<<"<<newGps.pitch<<endl;
              // cout<<setprecision(15)<<"newGps.roll<<"<<newGps.roll<<endl;
              fout_init_gps<<"%YAML:1.0"<<endl;
              fout_init_gps<<"# 建图时第一个点对应的GPS数据"<<endl;
              fout_init_gps<<setprecision(15)<<"OriLon: "<<newGps.lon<<endl;
              fout_init_gps<<setprecision(15)<<"OriLat: "<<newGps.lat<<endl;
              fout_init_gps<<setprecision(15)<<"OriAlt: "<<newGps.alt<<endl;
              fout_init_gps<<setprecision(15)<<"OriYaw: "<<newGps.heading<<endl;
              fout_init_gps<<setprecision(15)<<"OriPitch: "<<newGps.pitch<<endl;
              fout_init_gps<<setprecision(15)<<"OriRoll: "<<newGps.roll<<endl;

              fout_init_gps<<"# 当前车的GPS->LIDAR的外参，定位时融合GPS时要设置精确值，否则异常"<<endl;
              fout_init_gps<<setprecision(10)<<"compensate_init_yaw: "<<yaw_G2L<<endl;
              fout_init_gps<<setprecision(10)<<"compensate_init_pitch: "<<pitch_G2L<<endl;
              fout_init_gps<<setprecision(10)<<"compensate_init_roll: "<<roll_G2L<<endl;

              fout_init_gps<<"# 这个数字取决于建图时是否融合GPS，如果建图时融合了GPS数据，则与建图节点的yaw_G2L相等"<<endl;
              fout_init_gps<<setprecision(10)<<"mappingCarYawPara: "<<yaw_G2L<<endl;

              // fout_init_gps.close();

              // LLtoUTM(newGps.lat, newGps.lon, init_fix_odom_y, init_fix_odom_x, init_fix_zoom);

              UTMUPS::Forward(newGps.lat, newGps.lon, izone, northp, init_fix_odom_x, init_fix_odom_y);

              init_fix_odom_z = newGps.alt;

              // 我们车上使用的星网宇达GPS接收器，他的航向角方向与笛卡尔直角坐标系相反
              InitEulerAngle=Eigen::Vector3d(-((-newGps.heading+90.0)*deg + yaw_G2L), -(newGps.roll*deg), -(-newGps.pitch*deg));
              init_fix_odom_yaw = (-newGps.heading+90.0)*deg + yaw_G2L;
              // init_fix_odom_yaw = (-newGps.heading+90.0)*deg;
              rollAngle = (AngleAxisd(InitEulerAngle(2),Vector3d::UnitX()));
              pitchAngle = (AngleAxisd(InitEulerAngle(1),Vector3d::UnitY()));
              yawAngle = (AngleAxisd(InitEulerAngle(0),Vector3d::UnitZ()));
              init_fix_odom_pose = pitchAngle * yawAngle * rollAngle;
              // init_fix_odom_pose = yawAngle;
              // init_fix_odom_pose = init_fix_odom_pose.inverse();
              init_fix = true;
          }

          double northing, easting;
          std::string zone;

          UTMUPS::Forward(newGps.lat, newGps.lon, izone, northp, easting, northing);
          string zonestr = UTMUPS::EncodeZone(izone, northp);
          // 旧的地理坐标转换库
          // LLtoUTM(newGps.lat, newGps.lon, northing, easting, zone);

          PointType originateMGRS;
          //UTM->MGRS
          string mgrs;
          // int prec=5, 意思是精确到m，如果prec=7，精确到cm，如此类推
          MGRS::Forward(izone, northp, easting, northing, 7, mgrs);
          // mgrs code转成MGRS坐标
          // cout << stod(mgrs.substr(5, 7).insert(5, ".")) << " " <<stod(mgrs.substr(12).insert(5, ".")) << "\n";
          originateMGRS.x = stod(mgrs.substr(5, 7).insert(5, "."));
          originateMGRS.y = stod(mgrs.substr(12).insert(5, "."));
          OriginateGPStoMGRSforTest->push_back(originateMGRS);

          nav_msgs::Odometry fix_odom;
          fix_odom.header.stamp = newGps.header.stamp;
          fix_odom.header.frame_id = "camera_init";
          fix_odom.child_frame_id = "fix_utm";

          Eigen::Vector3d fix_odom_position;
          fix_odom_position.x() = easting - init_fix_odom_x;
          fix_odom_position.y() = northing - init_fix_odom_y;
          fix_odom_position.z() = newGps.alt - init_fix_odom_z;

          // fix_odom_position = fix_odom_position;
          // cout<<"init_fix_odom_pose: "<<init_fix_odom_pose<<endl;
          // 将GPS的position数据转到雷达坐标系
          fix_odom_position = yawAngle * fix_odom_position;
          // cout<<"fix_odom_position: "<<fix_odom_position.transpose()<<endl;

          fix_odom.pose.pose.position.x = fix_odom_position.y();
          fix_odom.pose.pose.position.y = fix_odom_position.z();
          fix_odom.pose.pose.position.z = fix_odom_position.x();
          
          Eigen::Vector3d CurreulerAngle((-newGps.heading+90.0)*deg + yaw_G2L, newGps.roll*deg, -newGps.pitch*deg);
          Eigen::Quaterniond tmp_q = AngleAxisd(CurreulerAngle(2),Vector3d::UnitX()) *
                                      AngleAxisd(CurreulerAngle(1),Vector3d::UnitY()) *
                                      AngleAxisd(CurreulerAngle(0),Vector3d::UnitZ());
          tmp_q = init_fix_odom_pose*tmp_q;
          fix_odom.pose.pose.orientation.x = tmp_q.y();
          fix_odom.pose.pose.orientation.y = tmp_q.z();
          fix_odom.pose.pose.orientation.z = tmp_q.x();
          fix_odom.pose.pose.orientation.w = tmp_q.w();

          fix_odom_pub.publish(fix_odom);

          PointType gps_position;
          gps_position.x = fix_odom.pose.pose.position.x;
          gps_position.y = fix_odom.pose.pose.position.y;
          gps_position.z = fix_odom.pose.pose.position.z;

          Eigen::Vector3d diffBetLidarAndGps;
          diffBetLidarAndGps.x() = transformAftMapped[5] - gps_position.z;
          diffBetLidarAndGps.y() = transformAftMapped[3] - gps_position.x;
          diffBetLidarAndGps.z() = transformAftMapped[4] - gps_position.y;
          double disGps2Lidar = diffBetLidarAndGps.norm();

          // mark the unreliable point for visualization
          // cout<<"status value: "<<newGps.status<<endl;
          // 0：初始化 48
          // 1：粗对准 
          // 2：精对准
          // 3：GPS定位 51
          // 4：GPS定向 52
          // 5：RTK    53
          // 6：DMI组合
          // 7：DMI标定
          // 8：纯惯性 56
          // 9：零速校正 57
          // A：VG模式 65
          // B：差分定向 66
          // C：动态对准 67
          // 这里需要设计专家系统来调整协方差
          // cout<<"newGps.status: "<<newGps.status<<endl;
          switch(newGps.status)
          {
              case 'B':
              {
                  gps_noise_x = 0.5;
                  gps_noise_y = 0.5;
                  gps_noise_z = 0.5;

                  gps_noise_att = 0.01;
                  indoor = false;
                  // cout<<"status is good!"<<endl;
                  gps_position.intensity = -255;
                  break;
              }
              case '5':
              {
                  gps_noise_x = 0.5;
                  gps_noise_y = 0.5;
                  gps_noise_z = 0.5;

                  gps_noise_att = 0.01;
                  indoor = false;
                  // cout<<"status is not so good!"<<endl;
                  gps_position.intensity = -255;
                  break;
              }
              case '4':
              {
                  gps_noise_x = 5.0;
                  gps_noise_y = 5.0;
                  gps_noise_z = 15.0;

                  gps_noise_att = 0.01;
                  indoor = false;
                  // cout<<"status is not so good!"<<endl;
                  gps_position.intensity = -255;
                  break;
              }
              case '3':
              {
                  gps_noise_x = 5.0;
                  gps_noise_y = 5.0;
                  gps_noise_z = 15.0;

                  gps_noise_att = 0.01;
                  indoor = false;
                  // cout<<"status is not so good!"<<endl;
                  gps_position.intensity = -255;
                  break;
              }
              default:
              {
                  indoor = true;
                  buffNum++;
                  if(buffNum > 100) buffNum = 100;

                  // gps_noise_x = 300.0;
                  // gps_noise_y = 300.0;

                  // gps_noise_x = buffNum*3;
                  // gps_noise_y = gps_noise_x;
                  gps_noise_x = 60.0;
                  gps_noise_y = gps_noise_x;

                  gps_noise_z = 5.0;
                  gps_noise_att = 0.1;
                  // cout<<"status is bad!"<<endl;
                  gps_position.intensity = 255;
                  break;
              }
          }

          // 如果接收到状态良好的GPS，但是不久前还是较差的信号
          if(!indoor && buffNum>0)
          {
              // 削弱室内状态认定
              buffNum--;
              if(buffNum<=0) buffNum=0;
              // 依然认为是在室内
              // indoor = true;
              
              // gps_noise_x = buffNum*3+0.1;
              gps_noise_x = 30.0;
              gps_noise_y = gps_noise_x;

              gps_noise_z = 5.0; 
              gps_noise_att = 0.1;

              gps_position.intensity = 255;
          }
          // cout<<"gps_noise_x: "<<gps_noise_x<<endl;

          OdomUTMPos = Eigen::Vector3d(fix_odom_position);

          if(useGPSfactor)
          {
            gtsam::noiseModel::Diagonal::shared_ptr GPS_Pose3Noise = 
              // noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << gps_noise_att, gps_noise_att, gps_noise_att, gps_noise_x, gps_noise_y, gps_noise_z).finished());
              noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.01, 0.01, 0.01, gps_noise_x, gps_noise_y, gps_noise_z).finished());
            
            // 鲁棒核函数
            // 参考 https://gtsam.org/2019/09/20/robust-noise-model.html
            auto huber = noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(5.0), GPS_Pose3Noise);
            
            gtsam::PriorFactor<gtsam::Pose3> GPS_Pose3Factor(X(cloudKeyPoses3D->points.size()), 
                                                            gtsam::Pose3(gtsam::Rot3::Quaternion(fix_odom.pose.pose.orientation.w,
                                                                                            fix_odom.pose.pose.orientation.z,
                                                                                            fix_odom.pose.pose.orientation.x,
                                                                                            fix_odom.pose.pose.orientation.y),
                                                                          gtsam::Point3(gps_position.z, gps_position.x, gps_position.y)), 
                                                                          huber);
            gtSAMgraph.add(GPS_Pose3Factor);

            LastOdomUTMPos = OdomUTMPos;

            // cout<<"add GPS!~~"<<cloudKeyPoses3D->points.size()<<endl;
          }
          // if(buffNum>=70 && useGPSfactor)
          // {
          //   gtsam::noiseModel::Diagonal::shared_ptr GPS_Pose3Noise = 
          //     noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << gps_noise_att, gps_noise_att, gps_noise_att, 300.0, 300.0, gps_noise_z).finished());
          //   gtsam::PriorFactor<gtsam::Pose3> GPS_Pose3Factor(X(cloudKeyPoses3D->points.size()), 
          //                                                   gtsam::Pose3(gtsam::Rot3::Quaternion(fix_odom.pose.pose.orientation.w,
          //                                                                                   fix_odom.pose.pose.orientation.z,
          //                                                                                   fix_odom.pose.pose.orientation.x,
          //                                                                                   fix_odom.pose.pose.orientation.y),
          //                                                                 gtsam::Point3(gps_position.z, gps_position.x, gps_position.y)), 
          //                                                                 GPS_Pose3Noise);
          //   gtSAMgraph.add(GPS_Pose3Factor);
          // }

          sleipnir_gps_queue.pop();

          // if(gps_position.intensity==255)
          // {
          //   // Define random generator with Gaussian distribution
          //   srand(cloudKeyPoses3D->points.size());  // 产生随机种子  把0换成NULL也行
          //   Eigen::Vector3d VecRand = Eigen::Vector3d::Random();
          //   // cout<<"VecRand: "<<VecRand.transpose()<<endl;
          //   SumRandom += VecRand;
          //   gps_position.x += SumRandom.x();
          //   gps_position.y += SumRandom.y();
          //   gps_position.z += SumRandom.z();
          // }

          fout_gps_evo<<timeLaserOdometry<<" "
                      <<gps_position.z<<" "
                      <<gps_position.x<<" "
                      <<gps_position.y<<" "
                      <<tmp_q.z()<<" "
                      <<tmp_q.x()<<" "
                      <<tmp_q.y()<<" "
                      <<tmp_q.w()<<endl;

          GPSHistoryPosition3D->push_back(gps_position);
          sensor_msgs::PointCloud2 gps_position_cloudMsgTemp;
          pcl::toROSMsg(*GPSHistoryPosition3D, gps_position_cloudMsgTemp);
          gps_position_cloudMsgTemp.header.stamp = newGps.header.stamp;
          gps_position_cloudMsgTemp.header.frame_id = "camera_init";
          fix_position_pub.publish(gps_position_cloudMsgTemp);
        }
      }
    }

    // gps odometry for kitti
    if (!customized_gps_msg) {
      while(!fix_queue.empty())
      {
        if(fix_queue.front().header.stamp.toSec()<timeLaserOdometry-0.02) fix_queue.pop();
        else
          break;
      }
      // while(!imu_raw_queue.empty())
      // {
      //   if(imu_raw_queue.front().header.stamp.toSec()<timeLaserOdometry-0.02) imu_raw_queue.pop();
      //   else
      //     break;
      // }
      if(!fix_queue.empty())
      {
          if(fix_queue.front().header.stamp.toSec()<timeLaserOdometry+0.02)
          {
            sensor_msgs::NavSatFix gps = fix_queue.front();

            if(!init_fix)
            {
                LLtoUTM(gps.latitude, gps.longitude, init_fix_odom_y, init_fix_odom_x, init_fix_zoom);
                init_fix_odom_z = gps.altitude;
                init_fix = true;
            }

            // while(!imu_raw_queue.empty())
            // {
            //   if(imu_raw_queue.front().header.stamp.toSec()<timeLaserOdometry) imu_raw_queue.pop();
            //   else
            //     break;
            // }
            // sensor_msgs::Imu imu_raw = imu_raw_queue.front();
            // Eigen::Quaterniond imu_raw_att(imu_raw.orientation.w,
            //                                 imu_raw.orientation.x,
            //                                 imu_raw.orientation.y,
            //                                 imu_raw.orientation.z);
            // imu_raw_att = first_imu_raw_att * imu_raw_att;

            double northing, easting;
            std::string zone;

            LLtoUTM(gps.latitude, gps.longitude, northing, easting, zone);
            nav_msgs::Odometry fix_odom;
            fix_odom.header.stamp = gps.header.stamp;

            fix_odom.header.frame_id = "camera_init";
            fix_odom.child_frame_id = "fix_utm";

            Eigen::Vector3d fix_odom_position;
            fix_odom_position.x() = easting - init_fix_odom_x;
            fix_odom_position.y() = northing - init_fix_odom_y;
            fix_odom_position.z() = gps.altitude - init_fix_odom_z;

            // fix_odom_position = fix_odom_position;
            // cout<<"init_fix_odom_pose: "<<init_fix_odom_pose<<endl;
            fix_odom_position = init_fix_odom_pose * fix_odom_position;

            fix_odom.pose.pose.position.x = fix_odom_position.y();
            fix_odom.pose.pose.position.y = fix_odom_position.z();
            fix_odom.pose.pose.position.z = fix_odom_position.x();
            
            fix_odom.pose.pose.orientation.x = 0;
            fix_odom.pose.pose.orientation.y = 0;
            fix_odom.pose.pose.orientation.z = 0;
            fix_odom.pose.pose.orientation.w = 1;

            fix_odom_pub.publish(fix_odom);

            PointType gps_position;
            gps_position.x = fix_odom.pose.pose.position.x;
            gps_position.y = fix_odom.pose.pose.position.y;
            gps_position.z = fix_odom.pose.pose.position.z;

            // 获取当前gps和tfm的位置
            // curr_gps << gps_position.x, gps_position.y, gps_position.z;
            // curr_tfm << transformAftMapped[3], transformAftMapped[4],transformAftMapped[5];

          }
      }

    }

    isam->update(gtSAMgraph, initialEstimate);
    isam->update();

    gtSAMgraph.resize(0);
    initialEstimate.clear();

    PointType thisPose3D;
    PointTypePose thisPose6D;
    Pose3 latestEstimate;

    isamCurrentEstimate = isam->calculateEstimate();
    latestEstimate = isamCurrentEstimate.at<Pose3>(X(cloudKeyPoses3D->points.size()));

    // // // Overwrite the beginning of the preintegration for the next step.
    // prev_state = NavState(isamCurrentEstimate.at<Pose3>(X(cloudKeyPoses3D->points.size())),
    //                       isamCurrentEstimate.at<Vector3>(V(cloudKeyPoses3D->points.size())));
    // prev_bias = isamCurrentEstimate.at<imuBias::ConstantBias>(B(cloudKeyPoses3D->points.size()));
    // // Reset the preintegration object.
    // imuIntegratorImu_->resetIntegrationAndSetBias(prev_bias);


    thisPose3D.x = latestEstimate.translation().y();
    thisPose3D.y = latestEstimate.translation().z();
    thisPose3D.z = latestEstimate.translation().x();
    thisPose3D.intensity = cloudKeyPoses3D->points.size();
    cloudKeyPoses3D->push_back(thisPose3D);

    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity;
    thisPose6D.roll = latestEstimate.rotation().pitch();
    thisPose6D.pitch = latestEstimate.rotation().yaw();
    thisPose6D.yaw = latestEstimate.rotation().roll();
    thisPose6D.time = timeLaserOdometry;
    cloudKeyPoses6D->push_back(thisPose6D);

    if (cloudKeyPoses3D->points.size() > 1) {
      transformAftMapped[0] = latestEstimate.rotation().pitch();
      transformAftMapped[1] = latestEstimate.rotation().yaw();
      transformAftMapped[2] = latestEstimate.rotation().roll();
      transformAftMapped[3] = latestEstimate.translation().y();
      transformAftMapped[4] = latestEstimate.translation().z();
      transformAftMapped[5] = latestEstimate.translation().x();

      for (int i = 0; i < 6; ++i) {
        transformLast[i] = transformAftMapped[i];
        transformTobeMapped[i] = transformAftMapped[i];
      }
    }

    pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr thisOutlierKeyFrame(
        new pcl::PointCloud<PointType>());

    pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame);
    pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);
    pcl::copyPointCloud(*laserCloudOutlierLastDS, *thisOutlierKeyFrame);

    /* 
        Scan Context loop detector 
        - ver 1: using surface feature as an input point cloud for scan context (2020.04.01: checked it works.)
        - ver 2: using downsampled original point cloud (/full_cloud_projected + downsampling)
        */
    bool usingRawCloud = true;
    if( usingRawCloud ) { // v2 uses downsampled raw point cloud, more fruitful height information than using feature points (v1)
        pcl::PointCloud<PointType>::Ptr thisRawCloudKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudRawDS,  *thisRawCloudKeyFrame);
        scManager.makeAndSaveScancontextAndKeys(*thisRawCloudKeyFrame);
    }
    else { // v1 uses thisSurfKeyFrame, it also works. (empirically checked at Mulran dataset sequences)
        scManager.makeAndSaveScancontextAndKeys(*thisSurfKeyFrame); 
    }
    // scManager.makeAndSaveScancontextAndKeys(*thisSurfKeyFrame); 

    // 转置点云，用于保存含强度信息的点云地图
    // @TODU: BUG, unused
    if(SaveIntensity)
    {
      pcl::PointCloud<PointType>::Ptr thisRawKeyFrame(
          new pcl::PointCloud<PointType>());
      size_t cloudSize = laserCloudRawDS->points.size();
      PointType thisPoint;
      for(size_t i = 0; i < cloudSize; ++i)
      {
        // copy该点用于操作
        thisPoint.x = laserCloudRawDS->points[i].x;
        thisPoint.y = laserCloudRawDS->points[i].y;
        thisPoint.z = laserCloudRawDS->points[i].z;

        laserCloudRawDS->points[i].x = thisPoint.y;
        laserCloudRawDS->points[i].y = thisPoint.z;
        laserCloudRawDS->points[i].z = thisPoint.x;
      }
      *thisRawKeyFrame = *transformPointCloud(laserCloudRawDS, &thisPose6D);
      originCloudKeyFrames.push_back(thisRawKeyFrame);
    }

    // if(!livox_points_msgs.empty())
    // {
    //   double livox_time;
    //   // mtx.lock();
    //   // extract the current gps message
    //   // std::lock_guard<std::mutex> lock(mtx);
    //   while(!livox_points_msgs.empty())
    //   {
    //     if(livox_points_msgs.front().header.stamp.toSec()<timeLaserOdometry)
    //     {
    //       cout<<"delta time: "<<livox_points_msgs.front().header.stamp.toSec()-timeLaserOdometry<<endl;
    //       livox_points_msgs.pop();
    //     }
    //     else
    //       break;
    //   }
    //   // mtx.unlock();
    //   cout<<"livox_points_msgs size: "<<livox_points_msgs.size()<<endl;
    // }

    // sensor_msgs::PointCloud2 cloudMsgTemp;
    // pcl::toROSMsg(*thisRawKeyFrame, cloudMsgTemp);
    // cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
    // cloudMsgTemp.header.frame_id = "camera_init";
    // pubIcpKeyFrames.publish(cloudMsgTemp);

    cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
    surfCloudKeyFrames.push_back(thisSurfKeyFrame);
    outlierCloudKeyFrames.push_back(thisOutlierKeyFrame);
    // cout<<"laserCloudRawDS:"<<laserCloudRawDS->size()<<endl;

    // in.open("/home/kyle/ros/kyle_ws/src/lins-gps-iris/src/position.csv",ios::app); //ios::trunc表示在打开文件前将文件清空,由于是写入,文件不存在则创建
    // in.precision(30);
    // in<<timeLaserOdometry<<","<<thisPose3D.x<<","<<thisPose3D.y<<","<<thisPose3D.z<<"\n";
    // in.close();
  }

  void correctPoses() {
    if (aLoopIsClosed == true) {
      recentCornerCloudKeyFrames.clear();
      recentSurfCloudKeyFrames.clear();
      recentOutlierCloudKeyFrames.clear();

      // int numPoses = isamCurrentEstimate.size();
      int numPoses = cloudKeyPoses3D->points.size();
      for (int i = 0; i < numPoses; ++i) {
        cloudKeyPoses3D->points[i].x =
            isamCurrentEstimate.at<Pose3>(X(i)).translation().y();
        cloudKeyPoses3D->points[i].y =
            isamCurrentEstimate.at<Pose3>(X(i)).translation().z();
        cloudKeyPoses3D->points[i].z =
            isamCurrentEstimate.at<Pose3>(X(i)).translation().x();

        cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
        cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
        cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
        cloudKeyPoses6D->points[i].roll =
            isamCurrentEstimate.at<Pose3>(X(i)).rotation().pitch();
        cloudKeyPoses6D->points[i].pitch =
            isamCurrentEstimate.at<Pose3>(X(i)).rotation().yaw();
        cloudKeyPoses6D->points[i].yaw =
            isamCurrentEstimate.at<Pose3>(X(i)).rotation().roll();
      }

      aLoopIsClosed = false;
    }
  }

  void clearCloud() {
    laserCloudCornerFromMap->clear();
    laserCloudSurfFromMap->clear();
    laserCloudCornerFromMapDS->clear();
    laserCloudSurfFromMapDS->clear();
  }

  // void closeFile()
  // {
  //   in.close();
  // }

  int lidarCounter = 0;
  double duration_ = 0;
  void run() {
    // 确保点云帧和里程计数据时间戳是一致的
    // 这里是不是可以考虑用message_filfer来处理更简单?
    if (newLaserCloudCornerLast &&
        std::abs(timeLaserCloudCornerLast - timeLaserOdometry) < 0.005 &&
        newLaserCloudSurfLast &&
        std::abs(timeLaserCloudSurfLast - timeLaserOdometry) < 0.005 &&
        newLaserCloudOutlierLast &&
        std::abs(timeLaserCloudOutlierLast - timeLaserOdometry) < 0.005 &&
        newLaserOdometry) {
      newLaserCloudCornerLast = false;
      newLaserCloudSurfLast = false;
      newLaserCloudOutlierLast = false;
      newLaserOdometry = false;
      newLaserCloudRawLast = false;

      // cout<<"timeLaserOdometry: "<<timeLaserOdometry<<endl;
      // cout<<"timeLaserCloudRawLast: "<<timeLaserCloudRawLast<<endl;

      // 获取互斥体的锁
      // 要百度一下std::lock_guard和std::mutex::lock的区别
      std::lock_guard<std::mutex> lock(mtx);

      // 后端优化不能太频繁,每隔mappingProcessInterval=0.3s进行一次
      if (timeLaserOdometry - timeLastProcessing >= mappingProcessInterval) {
        TicToc ts_total;

        // 记录本次优化的时间戳
        timeLastProcessing = timeLaserOdometry;

        // 基于匀速模型，根据上次微调的结果和odometry这次与上次计算的结果，
        // 猜测一个新的世界坐标系的转换矩阵transformTobeMapped[6],也就是将新的odom位姿转到map位姿
        transformAssociateToMap();

        extractSurroundingKeyFrames();

        downsampleCurrentScan();

        scan2MapOptimization();

        saveKeyFramesAndFactor();

        // 若成功检测到闭环,则进行pose修正
        correctPoses();

        publishTF();

        publishXYZTF();

        publishKeyPosesAndFrames();

        clearCloud();

        double time_total = ts_total.toc();
        if (VERBOSE) {
          duration_ =
              (duration_ * lidarCounter + time_total) / (lidarCounter + 1);
          lidarCounter++;
          // std::cout << "Mapping: time: " << duration_ << std::endl;
        }
      }
    }
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "lego_loam");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  ROS_INFO("\033[1;32m---->\033[0m Map Optimization Started.");

  parameter::readParameters(pnh);

  MappingHandler mappingHandler(nh, pnh);

  std::thread loopthread(&MappingHandler::loopClosureThread, &mappingHandler);
  std::thread visualizeMapThread(&MappingHandler::visualizeGlobalMapThread,
                                 &mappingHandler);

  ros::Rate rate(200);
  while (ros::ok()) {
    ros::spinOnce();

    mappingHandler.run();

    rate.sleep();
  }

  loopthread.join();
  visualizeMapThread.join();

  // mappingHandler.closeFile();

  return 0;
}
