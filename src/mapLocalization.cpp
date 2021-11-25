// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
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
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.
// #include "utility.h"
#include <parameters.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PointStamped.h>

#include <pcl/registration/ndt.h>
#include <sleipnir_msgs/sensorgps.h>

#include <sensor_msgs/NavSatStatus.h>
#include <sensor_msgs/NavSatFix.h>
#include <gps_common/conversions.h>

#include <tf/transform_listener.h>

using namespace gtsam;
using namespace parameter;
using namespace std;
using namespace Eigen;
using namespace gps_common;
using namespace GeographicLib;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

#define CORNER_MAP 0
#define SURF_MAP 1

const float scanPeriod = 0.1;
const int systemDelay = 0;
const int imuQueLength = 200;
// const string imuTopic = "/imu/data";

// 大cube就是附近地图
// 大cube各边上小cube个数的一半
int laserCloudCenWidth = 20;
int laserCloudCenHeight = 20;
int laserCloudCenDepth = 10;
// 多少个小cube组成大cube
const int laserCloudWidth = 41;
const int laserCloudHeight = 41;
const int laserCloudDepth = 21;
// 大cube含有小cube的个数
const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth; //

float box_center = 25.0;
float box_width = 50.0;

class mapOptimization{

private:

    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;

    noiseModel::Diagonal::shared_ptr priorNoise;
    noiseModel::Diagonal::shared_ptr odometryNoise;
    noiseModel::Diagonal::shared_ptr constraintNoise;
    noiseModel::Diagonal::shared_ptr LidarNoise;

    ros::NodeHandle nh;

    ros::Publisher pubOdomAftMapped;
    ros::Publisher pubKeyPoses;

    ros::Publisher pubRecentKeyFrames;

    ros::Publisher pubLaserCloudFullLast;

    ros::Publisher fix_odom_pub;
    ros::Publisher fix_position_pub;

    ros::Subscriber subLaserCloudCornerLast;
    ros::Subscriber subLaserCloudSurfLast;
    ros::Subscriber subOutlierCloudLast;
    ros::Subscriber subLaserCloudFullLast;
    ros::Subscriber subLaserOdometry;
    ros::Subscriber subImu;
    ros::Subscriber rviz_sub;
    ros::Subscriber sleipnir_gps_sub;

    nav_msgs::Odometry odomAftMapped;
    tf::StampedTransform aftMappedTrans;
    tf::TransformBroadcaster tfBroadcaster;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> outlierCloudKeyFrames;

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

    // PointType(pcl::PointXYZI)的XYZI分别保存3个方向上的平移和一个索引(cloudKeyPoses3D->points.size())
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;

    //PointTypePose的XYZI保存和cloudKeyPoses3D一样的内容，另外还保存RPY角度以及一个时间值timeLaserOdometry
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    
    // 结尾有DS代表是downsize,进行过下采样
    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses;
    pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;
    pcl::PointCloud<PointType>::Ptr laserCloudFullLast;
    pcl::PointCloud<PointType>::Ptr laserCloudFullLastDS;

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
    double timeLaserCloudFullLast;
    double timeLaserOdometry;
    double timeLaserCloudOutlierLast;
    double timeLastGloalMapPublish;

    bool newLaserCloudCornerLast;
    bool newLaserCloudSurfLast;
    bool newLaserCloudFullLast;
    bool newLaserOdometry;
    bool newLaserCloudOutlierLast;

    float transformLast[6];

    /*************高频转换量**************/
    // odometry计算得到的到世界坐标系下的转移矩阵
    float transformSum[6];
    // 转移增量，只使用了后三个平移增量
    float transformIncre[6];

    /*************低频转换量*************/
    // 以起始位置为原点的世界坐标系下的转换矩阵（猜测与调整的对象）
    float transformTobeMapped[6];
    // 存放mapping之前的Odometry计算的世界坐标系的转换矩阵（注：低频量，不一定与transformSum一样）
//    用来存放上一次transformSum，并且用于计算两个关键帧估计
    float transformBefMapped[6];
    // 存放mapping之后的经过mapping微调之后的转换矩阵
    float transformAftMapped[6];
    
    int imuPointerFront;
    int imuPointerLast;

    double imuTime[imuQueLength];
    float imuRoll[imuQueLength];
    float imuPitch[imuQueLength];
    std::deque<sensor_msgs::Imu> imuQue;

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

    float cRoll, sRoll, cPitch, sPitch, cYaw, sYaw, tX, tY, tZ;
    float ctRoll, stRoll, ctPitch, stPitch, ctYaw, stYaw, tInX, tInY, tInZ;

    string Corner_map_path;
    string Surf_map_path;
    string Full_map_path;
    // double map2cam_yaw, map2cam_pitch, map2cam_roll;
    
    // 用于存储预构建的点云地图
    pcl::PointCloud<PointType>::Ptr Corner_map_cloud;
    pcl::PointCloud<PointType>::Ptr Surf_map_cloud;
    pcl::PointCloud<PointType>::Ptr Full_map_cloud;
    // points in every cube
    pcl::PointCloud<PointType>::Ptr laserCloudCornerArray[laserCloudNum];
    pcl::PointCloud<PointType>::Ptr laserCloudSurfArray[laserCloudNum];
    // 用于匹配的附近cube数5*5*5=125
    int laserCloudValidInd[125];
    int laserCloudSurroundInd[125];
    // 发布点云地图
    ros::Publisher map_pub;

    // 初始位姿估计，也就是车辆初始在地图中的位姿
    tf::Pose init_guess;
    tf::Pose init_guess_pcl_map;
    bool set_init_guess = false;
    bool init_ICP = true; 
    // static Eigen::Isometry3f scan_init_guess_T;

    pcl::PointCloud<PointType>::Ptr InitPointCloudMap;

    queue<sleipnir_msgs::sensorgps> sleipnir_gps_queue;
    bool init_rviz = false;
    bool init_fix = false;
    double init_fix_odom_x, init_fix_odom_y, init_fix_odom_z; 
    Eigen::Vector3d InitEulerAngle;
    Eigen::Quaterniond init_fix_odom_pose;
    std::string init_fix_zoom;
    // double compensate_init_yaw, compensate_init_pitch, compensate_init_roll;
    // double mappingCarYawPara;
    Eigen::AngleAxisd rollAngle;
    Eigen::AngleAxisd pitchAngle;
    Eigen::AngleAxisd yawAngle;
    pcl::PointCloud<PointType>::Ptr GPSHistoryPosition3D;
    //   GPS measurement origination in map
    // double OriLon = 0.0;
    // double OriLat = 0.0;
    // double OriAlt = 0.0;
    // double OriYaw = 0.0;
    // double OriPitch = 0.0;
    // double OriRoll = 0.0;

    double gps_noise_x, gps_noise_y, gps_noise_z;
    double angle_noise;
    bool indoor = false;
    unsigned int buffNum = 0;
    bool useBetweenFactor = true;

    // 先验状态 和 后验状态
    gtsam::Rot3 prior_rotation;
    gtsam::Point3 prior_point; 
    gtsam::Pose3 prior_pose;
    gtsam::Vector3 prior_velocity = gtsam::Vector3(0,0,0);
    gtsam::NavState prev_state;
    gtsam::NavState prop_state;
    gtsam::imuBias::ConstantBias prior_imu_bias; // assume zero initial bias
    gtsam::imuBias::ConstantBias prev_bias = prior_imu_bias;

    // IMU预积分的一些参数配置
    // boost::shared_ptr<PreintegratedCombinedMeasurements::Params> imu_params = PreintegratedCombinedMeasurements::Params::MakeSharedD(imuGravity);
    boost::shared_ptr<gtsam::PreintegrationParams> imu_params = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
    // IMU预积分器
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;
    // 初始速度噪声模型
    noiseModel::Diagonal::shared_ptr velocity_noise_model = noiseModel::Isotropic::Sigma(3,1e8); // m/s
    noiseModel::Diagonal::shared_ptr bias_noise_model = noiseModel::Isotropic::Sigma(6,1e-3);

    double lastImuTime, curImuTime;

    bool useGPSfactor = true;

    tf::TransformListener pclmap_2_planning_odom_listener;

    int GpsStatusTest = 0;
    
    string rviz_frame = "_unset_rviz_map_frame";

public:

    mapOptimization():
        nh("~")
    {
        // 用于闭环图优化的参数设置，使用gtsam库
    	ISAM2Params parameters;
		parameters.relinearizeThreshold = 0.01;
		parameters.relinearizeSkip = 1;
    	isam = new ISAM2(parameters);

        pubKeyPoses = nh.advertise<sensor_msgs::PointCloud2>("/key_pose_origin", 2);
        pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> ("/aft_mapped_to_init", 5);

        pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/recent_cloud", 2);

        pubLaserCloudFullLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_full_last", 2);

        fix_odom_pub = nh.advertise<nav_msgs::Odometry>("/fix_odom", 5);
        fix_position_pub = nh.advertise<sensor_msgs::PointCloud2>("/gps_history_position", 2);

        // 设置滤波时创建的体素大小为0.2m/0.4m立方体,下面的单位为m
        downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilterSurf.setLeafSize(0.4, 0.4, 0.4);
        downSizeFilterOutlier.setLeafSize(0.4, 0.4, 0.4);

        downSizeFilterHistoryKeyFrames.setLeafSize(0.4, 0.4, 0.4);
        downSizeFilterSurroundingKeyPoses.setLeafSize(1.0, 1.0, 1.0);

        downSizeFilterGlobalMapKeyPoses.setLeafSize(1.0, 1.0, 1.0);
        downSizeFilterGlobalMapKeyFrames.setLeafSize(0.4, 0.4, 0.4);

        odomAftMapped.header.frame_id = "camera_init";
        odomAftMapped.child_frame_id = "aft_mapped";

        aftMappedTrans.frame_id_ = "camera_init";
        aftMappedTrans.child_frame_id_ = "aft_mapped";

        // 完整地图
        nh.param<string>("Corner_map_path", Corner_map_path, "Corner_map.pcd");
        // 纯平面点地图
        nh.param<string>("Surf_map_path", Surf_map_path, "Surf_map.pcd");
        // 纯边缘点地图
        nh.param<string>("Full_map_path", Full_map_path, "Full_map.pcd");
        // nh.param<double>("map2cam_yaw", map2cam_yaw, 0.0);
        // nh.param<double>("map2cam_pitch", map2cam_pitch, 0.0);
        // nh.param<double>("map2cam_roll", map2cam_roll, 0.0);

        // nh.param<double>("/OriLon", OriLon, 113.387985229);
        // nh.param<double>("/OriLat", OriLat, 23.040807724);
        // nh.param<double>("/OriAlt", OriAlt, 2.96000003815);
        // nh.param<double>("/OriYaw", OriYaw, 76.1139984131);
        // nh.param<double>("/OriPitch", OriPitch, -1.33500003815);
        // nh.param<double>("/OriRoll", OriRoll, 1.82000005245);
        // nh.param<double>("/compensate_init_yaw", compensate_init_yaw, 0.0);
        // nh.param<double>("/compensate_init_pitch", compensate_init_pitch, 0.0);
        // nh.param<double>("/compensate_init_roll", compensate_init_roll, 0.0);
        // nh.param<double>("/mappingCarYawPara", mappingCarYawPara, 0.0);
        nh.param<bool>("/useBetweenFactor", useBetweenFactor, true);
        nh.param<bool>("/useGPSfactor", useGPSfactor, true);
        // 实例化地图发布
        map_pub = nh.advertise<sensor_msgs::PointCloud2>("/map_pub_topic", 100);

        allocateMemory();

        // 等待3秒用于启动rviz，否则地图不能正常显示
        sleep(3);

        // 从pcd文件导入预构建的纯边缘点地图
        if(!LOAD_MAP(Corner_map_path, Corner_map_cloud))
        {
            ROS_WARN("Loading corner map from PCD file failed......");
            return;
        }

        // 从pcd文件导入预构建的纯平面点地图
        if(!LOAD_MAP(Surf_map_path, Surf_map_cloud))
        {
            ROS_WARN("Loading surface map from PCD file failed......");
            return;
        }

        for (int i = 0; i < laserCloudNum; i++)
        {
            laserCloudCornerArray[i].reset(new pcl::PointCloud<PointType>());
            laserCloudSurfArray[i].reset(new pcl::PointCloud<PointType>());
        }

        // 将两种地图分隔成栅格
        map2cube(CORNER_MAP, Corner_map_cloud);
        map2cube(SURF_MAP, Surf_map_cloud);

        subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2, &mapOptimization::laserCloudCornerLastHandler, this);
        subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2, &mapOptimization::laserCloudSurfLastHandler, this);
        subOutlierCloudLast = nh.subscribe<sensor_msgs::PointCloud2>("/outlier_cloud_last", 2, &mapOptimization::laserCloudOutlierLastHandler, this);
        subLaserCloudFullLast = nh.subscribe<sensor_msgs::PointCloud2>(LIDAR_TOPIC, 2, &mapOptimization::laserCloudFullLastHandler, this);
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 5, &mapOptimization::laserOdometryHandler, this);
        subImu = nh.subscribe<sensor_msgs::Imu> (IMU_TOPIC, 100, &mapOptimization::imuHandler, this);
        sleipnir_gps_sub = nh.subscribe("/sensorgps", 100, &mapOptimization::GpsHancdler, this);
        // 从rviz获取初始位姿
        rviz_sub = nh.subscribe("/initialpose", 1, &mapOptimization::rviz_cb, this);
    }

    void allocateMemory(){

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

        laserCloudFullLast.reset(new pcl::PointCloud<PointType>());
        laserCloudFullLastDS.reset(new pcl::PointCloud<PointType>());

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

        Corner_map_cloud.reset(new pcl::PointCloud<PointType>());
        Surf_map_cloud.reset(new pcl::PointCloud<PointType>());
        Full_map_cloud.reset(new pcl::PointCloud<PointType>());

        InitPointCloudMap.reset(new pcl::PointCloud<PointType>());

        GPSHistoryPosition3D.reset(new pcl::PointCloud<PointType>());

        timeLaserCloudCornerLast = 0;
        timeLaserCloudSurfLast = 0;
        timeLaserOdometry = 0;
        timeLaserCloudOutlierLast = 0;
        timeLastGloalMapPublish = 0;

        timeLastProcessing = -1;

        newLaserCloudCornerLast = false;
        newLaserCloudSurfLast = false;

        newLaserOdometry = false;
        newLaserCloudOutlierLast = false;

        for (int i = 0; i < 6; ++i){
            transformLast[i] = 0;
            transformSum[i] = 0;
            transformIncre[i] = 0;
            transformTobeMapped[i] = 0;
            transformBefMapped[i] = 0;
            transformAftMapped[i] = 0;
        }

        imuPointerFront = 0;
        imuPointerLast = -1;

        for (int i = 0; i < imuQueLength; ++i){
            imuTime[i] = 0;
            imuRoll[i] = 0;
            imuPitch[i] = 0;
        }

        // 噪声模型的赋值顺序参考 https://github.com/borglab/gtsam/issues/205
        // roll, pitch, yaw, x,y,z.
        gtsam::Vector Vector6(6);
        gtsam::Vector OdometryVector6(6);
        // Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;
        Vector6 << 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0;
        priorNoise = noiseModel::Diagonal::Variances(Vector6);
        OdometryVector6 << 1e-3, 1e-4, 1e-4, 1e-5, 1e-5, 1e-3;
        odometryNoise = noiseModel::Diagonal::Variances(OdometryVector6);

        gtsam::Vector LidarVector6(6);
        LidarVector6 << 1e-2, 1e-2, 1e-2, 1e-6, 1e-6, 1e-6;
        LidarNoise = noiseModel::Diagonal::Variances(LidarVector6);

        matA0 = cv::Mat (5, 3, CV_32F, cv::Scalar::all(0));
        matB0 = cv::Mat (5, 1, CV_32F, cv::Scalar::all(-1));
        matX0 = cv::Mat (3, 1, CV_32F, cv::Scalar::all(0));

        // matA1为边缘特征的协方差矩阵
        matA1 = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0));
        // matA1的特征值
        matD1 = cv::Mat (1, 3, CV_32F, cv::Scalar::all(0));
        // matA1的特征向量，对应于matD1存储
        matV1 = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0));

        isDegenerate = false;
        matP = cv::Mat (6, 6, CV_32F, cv::Scalar::all(0));

        laserCloudCornerFromMapDSNum = 0;
        laserCloudSurfFromMapDSNum = 0;
        laserCloudCornerLastDSNum = 0;
        laserCloudSurfLastDSNum = 0;
        laserCloudOutlierLastDSNum = 0;
        laserCloudSurfTotalLastDSNum = 0;

        potentialLoopFlag = false;

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

    // 应用LOAM的方法的话既要纯边缘点地图，也要纯平面点地图，因此要载入两个
    bool LOAD_MAP(string map_path, pcl::PointCloud<PointType>::Ptr map_cloud)
    {
        ROS_INFO("Loading map ...");
        Eigen::Affine3f map2cam = Eigen::Affine3f::Identity();
        // map2cam.rotate (Eigen::AngleAxisf(map2cam_yaw, Eigen::Vector3f::UnitZ())*
        //                 Eigen::AngleAxisf(map2cam_pitch, Eigen::Vector3f::UnitY())*
        //                 Eigen::AngleAxisf(map2cam_roll, Eigen::Vector3f::UnitX()));

        // 从文件导入指定pcd文件点云地图
        // pcl::io::loadPCDFile<PointType> (map_path, *map_cloud);
        pcl::io::loadPCDFile(map_path, *map_cloud);

        std::vector<int> indices; //保存去除的点的索引
        pcl::removeNaNFromPointCloud(*map_cloud,*map_cloud, indices); //去除点云中的NaN点

        // pcl::transformPointCloud(*map_cloud, *map_cloud, map2cam);

        // cout<<"map_cloud->size() = "<<map_cloud->size()<<endl;
        ROS_INFO("map_cloud->size() = %lu", map_cloud->size());
        // 检测文件是否有效
        if (map_cloud->size()==0){
            ROS_INFO("Map is empty...");
            return false;
        }

        return true;
    }

    void map2cube(int map_type, pcl::PointCloud<PointType>::Ptr map_cloud)
    {
        if(map_cloud->empty())
        {
            ROS_INFO("Map is empty...");
            return;
        }
        else
        {
            int map_cloud_num = map_cloud->size();
            for(int i=0;i<map_cloud_num;i++)
            {
                auto p = map_cloud->points[i];
                // 将点分配到对应的cube
                int cubeI = int((p.x + box_center) / box_width) + laserCloudCenWidth;
                int cubeJ = int((p.y + box_center) / box_width) + laserCloudCenHeight;
                int cubeK = int((p.z + box_center) / box_width) + laserCloudCenDepth;

                if (p.x + box_center < 0)
                    cubeI--;
                if (p.y + box_center < 0)
                    cubeJ--;
                if (p.z + box_center < 0)
                    cubeK--;
                
                if (cubeI >= 0 && cubeI < laserCloudWidth &&
                    cubeJ >= 0 && cubeJ < laserCloudHeight &&
                    cubeK >= 0 && cubeK < laserCloudDepth)
                {
                    int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
                    if(map_type == CORNER_MAP)
                    {
                        laserCloudCornerArray[cubeInd]->push_back(p);
                    }
                    else if(map_type == SURF_MAP)
                    {
                        laserCloudSurfArray[cubeInd]->push_back(p);
                    }
                }
            }
            ROS_INFO("%d Map rasterization is complete, number = %d\n", map_type, map_cloud_num);
        }
    }

    void visualizeGlobalMapThread() {
        ros::Rate rate(0.1);

        // 从pcd文件导入预构建的整点地图
        // if(!LOAD_MAP(Full_map_path, Full_map_cloud))
        // {
        //     ROS_WARN("Loading full map from PCD file failed......");
        //     return;
        // }
        *Full_map_cloud = *Corner_map_cloud + *Surf_map_cloud;
        // 每十秒发布一次全局点云地图，避免rviz屏蔽地图以后不能恢复
        ROS_INFO("Map has been published.");
        while (ros::ok()) {
            rate.sleep();
            // 提取地图中的点云，转为ros消息发布
            // 新建点云消息
            sensor_msgs::PointCloud2 map_msg;
            pcl::toROSMsg(*Full_map_cloud, map_msg);
            // 指定坐标系
            map_msg.header.frame_id = "camera_init";
            // 将地图发布出去才能在rviz可视化
            map_pub.publish(map_msg);
        }

    }

    // 将坐标转移到世界坐标系下,得到可用于建图的Lidar坐标，即修改了transformTobeMapped的值
//    https://zhuanlan.zhihu.com/p/159525107
    void transformAssociateToMap()
    {
//        transformAssociateToMap_();

        float x1 = cos(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) 
                 - sin(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);
        float y1 = transformBefMapped[4] - transformSum[4];
        float z1 = sin(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) 
                 + cos(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);

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

        float srx = -sbcx*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz)
                  - cbcx*sbcy*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                  - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                  - cbcx*cbcy*(calx*salz*(cblz*sbly - cbly*sblx*sblz) 
                  - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx);
        transformTobeMapped[0] = -asin(srx);

        float srycrx = sbcx*(cblx*cblz*(caly*salz - calz*salx*saly)
                     - cblx*sblz*(caly*calz + salx*saly*salz) + calx*saly*sblx)
                     - cbcx*cbcy*((caly*calz + salx*saly*salz)*(cblz*sbly - cbly*sblx*sblz)
                     + (caly*salz - calz*salx*saly)*(sbly*sblz + cbly*cblz*sblx) - calx*cblx*cbly*saly)
                     + cbcx*sbcy*((caly*calz + salx*saly*salz)*(cbly*cblz + sblx*sbly*sblz)
                     + (caly*salz - calz*salx*saly)*(cbly*sblz - cblz*sblx*sbly) + calx*cblx*saly*sbly);
        float crycrx = sbcx*(cblx*sblz*(calz*saly - caly*salx*salz)
                     - cblx*cblz*(saly*salz + caly*calz*salx) + calx*caly*sblx)
                     + cbcx*cbcy*((saly*salz + caly*calz*salx)*(sbly*sblz + cbly*cblz*sblx)
                     + (calz*saly - caly*salx*salz)*(cblz*sbly - cbly*sblx*sblz) + calx*caly*cblx*cbly)
                     - cbcx*sbcy*((saly*salz + caly*calz*salx)*(cbly*sblz - cblz*sblx*sbly)
                     + (calz*saly - caly*salx*salz)*(cbly*cblz + sblx*sbly*sblz) - calx*caly*cblx*sbly);
        transformTobeMapped[1] = atan2(srycrx / cos(transformTobeMapped[0]), 
                                       crycrx / cos(transformTobeMapped[0]));
        
        float srzcrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                     - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
                     - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                     - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                     + cbcx*sbcz*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz);
        float crzcrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                     - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                     - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                     - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
                     + cbcx*cbcz*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz);
        transformTobeMapped[2] = atan2(srzcrx / cos(transformTobeMapped[0]), 
                                       crzcrx / cos(transformTobeMapped[0]));

        x1 = cos(transformTobeMapped[2]) * transformIncre[3] - sin(transformTobeMapped[2]) * transformIncre[4];
        y1 = sin(transformTobeMapped[2]) * transformIncre[3] + cos(transformTobeMapped[2]) * transformIncre[4];
        z1 = transformIncre[5];

        x2 = x1;
        y2 = cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
        z2 = sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

        transformTobeMapped[3] = transformAftMapped[3] 
                               - (cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2);
        transformTobeMapped[4] = transformAftMapped[4] - y2;
        transformTobeMapped[5] = transformAftMapped[5] 
                               - (-sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2);

        // cout<<"transformTobeMapped: "<<endl;
        // for(int l=0;l<6;l++)
        // {
        //     cout<<transformTobeMapped[l]<<", ";
        // }
        // cout<<endl<<endl;

        currentRobotPosPoint.x = transformAftMapped[3];
        currentRobotPosPoint.y = transformAftMapped[4];
        currentRobotPosPoint.z = transformAftMapped[5];
    }
    
    void transformAssociateToMap_()
    {
//        transformTobeMapped
        Eigen::AngleAxisd rv_transformTobeMapped;
        rv_transformTobeMapped=Eigen::AngleAxisd(transformTobeMapped[2],Vector3d::UnitZ())*
                                Eigen::AngleAxisd(transformTobeMapped[0],Vector3d::UnitX())*
                                Eigen::AngleAxisd(transformTobeMapped[1],Vector3d::UnitY());
        Eigen::Quaterniond Q_transformTobeMapped = Eigen::Quaterniond(rv_transformTobeMapped);
        Eigen::Vector3d V_transformTobeMapped(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]);
//        cout<<"Q_transformTopped:"<<Q_transformTopped.coeffs().transpose()<<endl;

//        transformBefMapped
        Eigen::AngleAxisd rv_transformBefMapped;
        rv_transformBefMapped=Eigen::AngleAxisd(transformBefMapped[2],Vector3d::UnitZ())*
                               Eigen::AngleAxisd(transformBefMapped[0],Vector3d::UnitX())*
                               Eigen::AngleAxisd(transformBefMapped[1],Vector3d::UnitY());
        Eigen::Quaterniond Q_transformBefMapped = Eigen::Quaterniond(rv_transformBefMapped);
        Eigen::Vector3d V_transformBefMapped(transformBefMapped[3], transformBefMapped[4], transformBefMapped[5]);
//        cout<<"Q_transformBefMapped:"<<Q_transformBefMapped.coeffs().transpose()<<endl;

//        transformSum
        Eigen::AngleAxisd rv_transformSum;
        rv_transformSum=Eigen::AngleAxisd(transformSum[2],Vector3d::UnitZ())*
                              Eigen::AngleAxisd(transformSum[0],Vector3d::UnitX())*
                              Eigen::AngleAxisd(transformSum[1],Vector3d::UnitY());
        Eigen::Quaterniond Q_transformSum = Eigen::Quaterniond(rv_transformSum);
        Eigen::Vector3d V_transformSum(transformSum[3], transformSum[4], transformSum[5]);
//        cout<<"Q_transformSum:"<<Q_transformSum.coeffs().transpose()<<endl;

//        transformAftMapped
        Eigen::AngleAxisd rv_transformAftMapped;
        rv_transformAftMapped=Eigen::AngleAxisd(transformAftMapped[2],Vector3d::UnitZ())*
                        Eigen::AngleAxisd(transformAftMapped[0],Vector3d::UnitX())*
                        Eigen::AngleAxisd(transformAftMapped[1],Vector3d::UnitY());
        Eigen::Quaterniond Q_transformAftMapped = Eigen::Quaterniond(rv_transformAftMapped);
        Eigen::Vector3d V_transformAftMapped(transformAftMapped[3], transformAftMapped[4], transformAftMapped[5]);
//        cout<<"Q_transformAftMapped:"<<Q_transformAftMapped.coeffs().transpose()<<endl;

//      计算位姿变换
//        Eigen::Quaterniond diff_q_odom = Q_transformBefMapped.inverse() * Q_transformSum;
        Eigen::Vector3d diff_t_odom = Q_transformBefMapped.inverse()*(V_transformSum - V_transformBefMapped);

        Q_transformTobeMapped = Q_transformAftMapped * Q_transformBefMapped.inverse() * Q_transformSum;
        V_transformTobeMapped = V_transformAftMapped + Q_transformAftMapped * diff_t_odom;

//        geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw
//                (transformAftMapped[2], -transformAftMapped[0], -transformAftMapped[1]);
//        tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w)

        Eigen::Vector3d Angles_Tmp = Q_transformTobeMapped.toRotationMatrix().eulerAngles(1,0,2);
//        double transformTobeMapped_[6];
        transformTobeMapped[0] = Angles_Tmp.y();
        transformTobeMapped[1] = Angles_Tmp.x();
        transformTobeMapped[2] = Angles_Tmp.z();
        transformTobeMapped[3] = V_transformTobeMapped.x();
        transformTobeMapped[4] = V_transformTobeMapped.y();
        transformTobeMapped[5] = V_transformTobeMapped.z();

        // cout<<"transformTobeMapped: "<<endl;
        // for(int l=0;l<6;l++)
        // {
        //     cout<<transformTobeMapped[l]<<", ";
        // }
        // cout<<endl;
    }

    //记录odometry发送的转换矩阵与mapping之后的转换矩阵，下一帧点云会使用(有IMU的话会使用IMU进行补偿)
    // 更新transformBefMapped和transformTobeMapped
    void transformUpdate()
    {
		if (imuPointerLast >= 0) {
		    float imuRollLast = 0, imuPitchLast = 0;
		    //查找点云时间戳小于imu时间戳的imu位置
            while (imuPointerFront != imuPointerLast) {
		        if (timeLaserOdometry + scanPeriod < imuTime[imuPointerFront]) {
		            break;
		        }
		        imuPointerFront = (imuPointerFront + 1) % imuQueLength;
		    }

		    if (timeLaserOdometry + scanPeriod > imuTime[imuPointerFront]) {//未找到,此时imuPointerFront==imuPointerLast
		        imuRollLast = imuRoll[imuPointerFront];
		        imuPitchLast = imuPitch[imuPointerFront];
		    } else {
		        int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
		        float ratioFront = (timeLaserOdometry + scanPeriod - imuTime[imuPointerBack]) 
		                         / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
		        float ratioBack = (imuTime[imuPointerFront] - timeLaserOdometry - scanPeriod) 
		                        / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

		        //按时间比例求翻滚角和俯仰角
                imuRollLast = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
		        imuPitchLast = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
		    }

		    //imu稍微补偿俯仰角和翻滚角
            transformTobeMapped[0] = 0.998 * transformTobeMapped[0] + 0.002 * imuPitchLast;
		    transformTobeMapped[2] = 0.998 * transformTobeMapped[2] + 0.002 * imuRollLast;
		  }

		//记录优化之前与之后的转移矩阵
        for (int i = 0; i < 6; i++) {
		    transformBefMapped[i] = transformSum[i];
		    transformAftMapped[i] = transformTobeMapped[i];
		}
    }

    void updatePointAssociateToMapSinCos(){
        // 先提前求好roll,pitch,yaw的sin和cos值
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

    //根据调整计算后的转移矩阵，将点注册到全局世界坐标系下
    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        // 进行6自由度的变换，先进行旋转，然后再平移
        // 主要进行坐标变换，将局部坐标转换到全局坐标中去	

        // 先绕z轴旋转
        //     |cosrz  -sinrz  0|
        //  Rz=|sinrz  cosrz   0|
        //     |0       0      1|
        // [x1,y1,z1]^T=Rz*[pi->x,pi->y,pi->z]
        float x1 = cYaw * pi->x - sYaw * pi->y;
        float y1 = sYaw * pi->x + cYaw * pi->y;
        float z1 = pi->z;

        // [x2,y2,z2]^T=Rx*[x1,y1,z1]
        //    |1     0        0|
        // Rx=|0   cosrx -sinrx|
        //    |0   sinrx  cosrx|
        float x2 = x1;
        float y2 = cRoll * y1 - sRoll * z1;
        float z2 = sRoll * y1 + cRoll * z1;

        // 最后再绕Y轴旋转，然后加上平移
        //    |cosry   0   sinry|
        // Ry=|0       1       0|
        //    |-sinry  0   cosry|
        po->x = cPitch * x2 + sPitch * z2 + tX;
        po->y = y2 + tY;
        po->z = -sPitch * x2 + cPitch * z2 + tZ;
        po->intensity = pi->intensity;
    }

    void updateTransformPointCloudSinCos(PointTypePose *tIn){

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

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn){

        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;
        PointType pointTo;

        int cloudSize = cloudIn->points.size();
        cloudOut->resize(cloudSize);

        for (int i = 0; i < cloudSize; ++i){

            pointFrom = &cloudIn->points[i];
            float x1 = ctYaw * pointFrom->x - stYaw * pointFrom->y;
            float y1 = stYaw * pointFrom->x + ctYaw* pointFrom->y;
            float z1 = pointFrom->z;

            float x2 = x1;
            float y2 = ctRoll * y1 - stRoll * z1;
            float z2 = stRoll * y1 + ctRoll* z1;

            pointTo.x = ctPitch * x2 + stPitch * z2 + tInX;
            pointTo.y = y2 + tInY;
            pointTo.z = -stPitch * x2 + ctPitch * z2 + tInZ;
            pointTo.intensity = pointFrom->intensity;

            cloudOut->points[i] = pointTo;
        }
        return cloudOut;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn){

        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;
        PointType pointTo;

        int cloudSize = cloudIn->points.size();
        cloudOut->resize(cloudSize);

		// 坐标系变换，旋转rpy角
        for (int i = 0; i < cloudSize; ++i){

            pointFrom = &cloudIn->points[i];
            float x1 = cos(transformIn->yaw) * pointFrom->x - sin(transformIn->yaw) * pointFrom->y;
            float y1 = sin(transformIn->yaw) * pointFrom->x + cos(transformIn->yaw)* pointFrom->y;
            float z1 = pointFrom->z;

            float x2 = x1;
            float y2 = cos(transformIn->roll) * y1 - sin(transformIn->roll) * z1;
            float z2 = sin(transformIn->roll) * y1 + cos(transformIn->roll)* z1;

            pointTo.x = cos(transformIn->pitch) * x2 + sin(transformIn->pitch) * z2 + transformIn->x;
            pointTo.y = y2 + transformIn->y;
            pointTo.z = -sin(transformIn->pitch) * x2 + cos(transformIn->pitch) * z2 + transformIn->z;
            pointTo.intensity = pointFrom->intensity;

            cloudOut->points[i] = pointTo;
        }
        return cloudOut;
    }

    void laserCloudOutlierLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudOutlierLast = msg->header.stamp.toSec();
        laserCloudOutlierLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudOutlierLast);
        newLaserCloudOutlierLast = true;
    }

    void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudCornerLast = msg->header.stamp.toSec();
        laserCloudCornerLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudCornerLast);
        newLaserCloudCornerLast = true;
    }

    void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudSurfLast = msg->header.stamp.toSec();
        laserCloudSurfLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudSurfLast);
        newLaserCloudSurfLast = true;
    }

    void laserCloudFullLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudFullLast = msg->header.stamp.toSec();
        laserCloudFullLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudFullLast);
        std::vector<int> indices;
        // 去除nan点
        pcl::removeNaNFromPointCloud(*laserCloudFullLast, *laserCloudFullLast, indices);
        // tranform the laser point cloud from /vehicle to /camera
        for(int i=0;i<laserCloudFullLast->size();i++)
        {
            double tmp = laserCloudFullLast->points[i].x;
            laserCloudFullLast->points[i].x = laserCloudFullLast->points[i].y;
            laserCloudFullLast->points[i].y = laserCloudFullLast->points[i].z;
            laserCloudFullLast->points[i].z = tmp;
        }
        newLaserCloudFullLast = true;
    }

    // 这里实现了camera_init --> map
    void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr& laserOdometry){
        timeLaserOdometry = laserOdometry->header.stamp.toSec();
        double roll, pitch, yaw;
        geometry_msgs::Quaternion geoQuat = laserOdometry->pose.pose.orientation;
        tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w)).getRPY(roll, pitch, yaw);
        // 雷达里程计
        transformSum[0] = -pitch;
        transformSum[1] = -yaw;
        transformSum[2] = roll;
        transformSum[3] = laserOdometry->pose.pose.position.x;
        transformSum[4] = laserOdometry->pose.pose.position.y;
        transformSum[5] = laserOdometry->pose.pose.position.z;
        newLaserOdometry = true;
    }

    // sleipnir GPS
    void GpsHancdler(const sleipnir_msgs::sensorgps::ConstPtr &msg)
    {
        sleipnir_msgs::sensorgps currGps = *msg;
        // currGps.header.stamp.sec -= 12;
        sleipnir_gps_queue.push(currGps);

        // cout<<currGps<<endl<<endl;
        if(currGps.satenum <= 25 && !init_fix && (GpsStatusTest%100==0))
        {
            cout<<RED<<"当前卫星信号较差，请手动指定初始位置，或驾驶车辆到室外卫星信号良好区域重新启动！"<<RESET<<endl;
            GpsStatusTest++;
        }
        if(currGps.satenum > 25 && !init_fix)  // 当前星数大于35且还未初始化
        {
            // 这里应该从文件中读取地图起点处的GPS数据，然后将其转换为UTM的起点，每个在线数据都减去该起点坐标
            // LLtoUTM(OriLat, OriLon, init_fix_odom_y, init_fix_odom_x, init_fix_zoom);
            bool northp;
            int izone;
            UTMUPS::Forward(OriLat, OriLon, izone, northp, init_fix_odom_x, init_fix_odom_y);
            init_fix_odom_z = OriAlt;

            // 我们车上使用的星网宇达GPS接收器，他的航向角方向与笛卡尔直角坐标系相反
            InitEulerAngle=Eigen::Vector3d((-OriYaw+90.0)*deg+mappingCarYawPara, OriRoll*deg, -OriPitch*deg);
            rollAngle = (AngleAxisd(InitEulerAngle(2),Vector3d::UnitX()));
            pitchAngle = (AngleAxisd(InitEulerAngle(1),Vector3d::UnitY()));
            yawAngle = (AngleAxisd(InitEulerAngle(0),Vector3d::UnitZ()));
            // init_fix_odom_pose = rollAngle*pitchAngle*yawAngle;
            init_fix_odom_pose = pitchAngle * yawAngle * rollAngle;
            init_fix_odom_pose = init_fix_odom_pose.inverse();

            double northing, easting;
            std::string zone;

            // LLtoUTM(currGps.lat, currGps.lon, northing, easting, zone);
            UTMUPS::Forward(currGps.lat, currGps.lon, izone, northp, easting, northing);

            Eigen::Vector3d fix_odom_position;
            fix_odom_position.x() = easting - init_fix_odom_x;
            fix_odom_position.y() = northing - init_fix_odom_y;
            // 考虑到单GPS的海拔精度非常低，因此这里索性设为
            fix_odom_position.z() = currGps.alt - init_fix_odom_z;
            // fix_odom_position.z() = 0.0;

            // fix_odom_position = fix_odom_position;
            // cout<<"init_fix_odom_pose: "<<init_fix_odom_pose<<endl;
            fix_odom_position = yawAngle.inverse() * fix_odom_position;

            // fix_odom.pose.pose.position.x = fix_odom_position.y();
            // fix_odom.pose.pose.position.y = fix_odom_position.z();
            // fix_odom.pose.pose.position.z = fix_odom_position.x();

            // cout<<"GPS YAW："<<-newGps.heading+90.0+compensate_init_yaw*rad-InitEulerAngle(0)*rad<<endl;
            // cout<<"Fusion YAW："<<transformTobeMapped[1]*rad<<endl;
            
            Eigen::Vector3d CurreulerAngle((-currGps.heading+90.0)*deg+compensate_init_yaw, currGps.roll*deg, -currGps.pitch*deg);
            Eigen::Quaterniond tmp_q = AngleAxisd(CurreulerAngle(2),Vector3d::UnitX()) *
                                        AngleAxisd(CurreulerAngle(1),Vector3d::UnitY()) *
                                        AngleAxisd(CurreulerAngle(0),Vector3d::UnitZ());
            tmp_q = init_fix_odom_pose*tmp_q;
            // fix_odom.pose.pose.orientation.x = tmp_q.y();
            // fix_odom.pose.pose.orientation.y = tmp_q.z();
            // fix_odom.pose.pose.orientation.z = tmp_q.x();
            // fix_odom.pose.pose.orientation.w = tmp_q.w();

            // update the transformTobeMapped and the transformAftMapped
            transformAftMapped[3] = fix_odom_position.y();
            transformAftMapped[4] = fix_odom_position.z();
            transformAftMapped[5] = fix_odom_position.x();
            // cout<<"fix_odom_position: "<<fix_odom_position.transpose()<<endl;
            transformAftMapped[0] = CurreulerAngle[1] - InitEulerAngle[1];
            transformAftMapped[1] = CurreulerAngle[0] - InitEulerAngle[0];
            transformAftMapped[2] = CurreulerAngle[2] - InitEulerAngle[2];
            // cout<<"CurreulerAngle-InitEulerAngle: "<<(CurreulerAngle-InitEulerAngle).transpose()<<endl;

            // for (int i = 0; i < 6; ++i){
            // 	transformTobeMapped[i] = transformAftMapped[i];
            // }

            init_fix = true; // 确保不二次进入本次判断
            set_init_guess = true; // 告诉程序可以进行优化匹配
            init_ICP = false;

            cout<<GREEN<<"已通过GPS数据获取初始位置估计！"<<RESET<<endl;
        }
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

    // 只提取imu中的roll, pitch???
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn) {
        std::lock_guard<std::mutex> lock(mtx);
        double roll, pitch, yaw;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(imuIn->orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        imuPointerLast = (imuPointerLast + 1) % imuQueLength;
        imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
        imuRoll[imuPointerLast] = roll;
        imuPitch[imuPointerLast] = pitch;

        sensor_msgs::Imu thisImu = *imuIn;
        imuQue.push_back(thisImu);
    }

    void rviz_cb(geometry_msgs::PoseWithCovarianceStamped rviz_msg)
    {
        ROS_INFO("Set initial pose estimation from rviz......");
        tf::Stamped<tf::Pose> init_guess_stamp;
        geometry_msgs::PoseStamped msg_init_guess;
        geometry_msgs::PoseStamped msg_init_guess_pcl_map;

        // 从消息中提取初始姿态信息
        // 后面还需要改进，这里应该是动态的，根据消息的header监听不同的tf
        // tf::poseMsgToTF(rviz_msg.pose.pose, init_guess);

        cout<<"frame_id: "<<rviz_msg.header.frame_id<<endl;
        rviz_frame = rviz_msg.header.frame_id;

        // if(rviz_msg.header.frame_id == "planning_odom")
        // {
        //     msg_init_guess.header = rviz_msg.header;
        //     msg_init_guess.pose = rviz_msg.pose.pose;
        //     pclmap_2_planning_odom_listener.transformPose("pcl_map", msg_init_guess, msg_init_guess_pcl_map);

        //     tf::poseStampedMsgToTF(msg_init_guess_pcl_map, init_guess_stamp);
        // }
        // else if(rviz_msg.header.frame_id == "pcl_map")
        // {
        //     msg_init_guess_pcl_map.header = rviz_msg.header;
        //     msg_init_guess_pcl_map.pose = rviz_msg.pose.pose;
        //     tf::poseStampedMsgToTF(msg_init_guess_pcl_map, init_guess_stamp);
        // }

        if(rviz_msg.header.frame_id == "pcl_map")
        {
            msg_init_guess_pcl_map.header = rviz_msg.header;
            msg_init_guess_pcl_map.pose = rviz_msg.pose.pose;
            tf::poseStampedMsgToTF(msg_init_guess_pcl_map, init_guess_stamp);
        }
        else
        {
            msg_init_guess.header = rviz_msg.header;
            msg_init_guess.pose = rviz_msg.pose.pose;
            pclmap_2_planning_odom_listener.transformPose("pcl_map", msg_init_guess, msg_init_guess_pcl_map);

            tf::poseStampedMsgToTF(msg_init_guess_pcl_map, init_guess_stamp);
        }

        // cout<<"init_guess_pcl_map z : "<<gm_init_guess_pcl_map.pose.position.z<<endl;

        // transform to /camera_init
        tf::Vector3 init_guess_t;
        init_guess_t.setX(init_guess_stamp.getOrigin().getY());
        init_guess_t.setY(init_guess_stamp.getOrigin().getZ());
        init_guess_t.setZ(init_guess_stamp.getOrigin().getX());
        // 获取旋转
        double roll, pitch, yaw;
//        tf::Matrix3x3(init_guess.getRotation()).getEulerYPR(roll, yaw, pitch);
        tf::Matrix3x3(init_guess_stamp.getRotation()).getEulerYPR(pitch, roll, yaw);

        cout<<"Initial position:    x="<<init_guess_t.getX()<<", y="<<init_guess_t.getY()<<", z="<<init_guess_t.getZ()<<endl;
        cout<<"Initial_orientation: yaw="<<yaw/PI*180<<", pitch="<<pitch/PI*180<<", roll="<<roll/PI*180<<endl;

        ROS_INFO("Set initial pose successfully......\n");
        // 重置位姿初值
//         q_w_curr.w() = init_guess.getRotation().w();
//         q_w_curr.x() = init_guess.getRotation().x();
//         q_w_curr.y() = init_guess.getRotation().y();
//         q_w_curr.z() = init_guess.getRotation().z();
//         t_w_curr.x() = init_guess.getOrigin().getX();
//         t_w_curr.y() = init_guess.getOrigin().getY();
//         t_w_curr.z() = init_guess.getOrigin().getZ();
        transformAftMapped[0] = roll;
        transformAftMapped[1] = pitch;
        transformAftMapped[2] = yaw;
        transformAftMapped[3] = init_guess_t.getX();
        transformAftMapped[4] = init_guess_t.getY();
        transformAftMapped[5] = init_guess_t.getZ();

        // init_rviz = true;
        set_init_guess = true; // 告诉程序可以启动
        init_ICP = false; // 告诉程序需要进行一次ICP初始估计
    }

    // tf : camera_init --> camera
    void publishTF(){
        geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw
                                  (transformAftMapped[2], -transformAftMapped[0], -transformAftMapped[1]);

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
        aftMappedTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
        aftMappedTrans.setOrigin(tf::Vector3(transformAftMapped[3], transformAftMapped[4], transformAftMapped[5]));
        tfBroadcaster.sendTransform(aftMappedTrans);
    }

    void publishKeyPosesAndFrames(){

        if (pubKeyPoses.getNumSubscribers() != 0){
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*cloudKeyPoses3D, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "camera_init";
            pubKeyPoses.publish(cloudMsgTemp);
        }

        if (pubRecentKeyFrames.getNumSubscribers() != 0){
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*laserCloudSurfFromMapDS, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "camera_init";
            pubRecentKeyFrames.publish(cloudMsgTemp);
        }
    }

    Pose3 pclPointTogtsamPose3(PointTypePose thisPoint){
    	return Pose3(Rot3::RzRyRx(double(thisPoint.yaw), double(thisPoint.roll), double(thisPoint.pitch)),
                           Point3(double(thisPoint.z),   double(thisPoint.x),    double(thisPoint.y)));
    }

    Eigen::Affine3f pclPointToAffine3fCameraToLidar(PointTypePose thisPoint){
    	return pcl::getTransformation(thisPoint.z, thisPoint.x, thisPoint.y, thisPoint.yaw, thisPoint.roll, thisPoint.pitch);
    }

    void extractSurroundingKeyFrames(){

        // if (cloudKeyPoses3D->points.empty() == true)
        //     return;	

        // 每个cube为50m x 50m x 50m的立方体，+25是为了把当前位置移到最中心的cube的中心位置
        // 下面计算的centerCubeI,centerCubeJ，centerCubeK是一种索引，指明当前收到的点云所在的总cube的中心位置
        int centerCubeI = int((currentRobotPosPoint.x + box_center) / box_width) + laserCloudCenWidth;
        int centerCubeJ = int((currentRobotPosPoint.y + box_center) / box_width) + laserCloudCenHeight;
        int centerCubeK = int((currentRobotPosPoint.z + box_center) / box_width) + laserCloudCenDepth;

        //由于计算机求商是向零取整，为了不使（-box_width,box_width）范围内的数与box_width求商后都向零偏移，当被除数为负数时求商结果统一向左偏移一个单位，也即减一
        if (currentRobotPosPoint.x + box_center < 0)
            centerCubeI--;
        if (currentRobotPosPoint.y + box_center < 0)
            centerCubeJ--;
        if (currentRobotPosPoint.z + box_center < 0)
            centerCubeK--;

        // 注意这里的centerCube Index有可能为负数，为了下面搜索方便，下面作调整
        // 调整之后取值范围:3 < centerCubeI < 18， 3 < centerCubeJ < 8, 3 < centerCubeK < 18，因为要保证附近有可搜索的cube(前2，后2，中1)
        // 如果处于下边界，表明地图向负方向延伸的可能性比较大，则循环移位，将数组中心点向上边界调整一个单位
        // 接下来对做一些调整，确保位姿在cube中的相对位置（centerCubeI，centerCubeJ，centerCubeK）能够有一个5*5*5（前2个cube，后两个cube，中间1个cube组成5个？）的邻域。
        // 下面相当与挪动大cube的位置，激光雷达的位姿和map是不变的
        while (centerCubeI < 3)
        {
            for (int j = 0; j < laserCloudHeight; j++) // y轴方向
            {
                for (int k = 0; k < laserCloudDepth; k++) // z轴方向
                { 
                    int i = laserCloudWidth - 1;
                    // 暂存一个cube
                    pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k]; 
                    pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    for (; i >= 1; i--) // x轴方向
                    {
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudCornerArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudSurfArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    }
                    laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        laserCloudCubeCornerPointer;
                    laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        laserCloudCubeSurfPointer;
                    laserCloudCubeCornerPointer->clear();
                    laserCloudCubeSurfPointer->clear();
                }
            }

            centerCubeI++;
            laserCloudCenWidth++; // 用于待会匹配完后将点映射到cube相对位置
        }

        while (centerCubeI >= laserCloudWidth - 3)
        { 
            for (int j = 0; j < laserCloudHeight; j++)
            {
                for (int k = 0; k < laserCloudDepth; k++)
                {
                    int i = 0;
                    pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    for (; i < laserCloudWidth - 1; i++)
                    {
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudCornerArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudSurfArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    }
                    laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        laserCloudCubeCornerPointer;
                    laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        laserCloudCubeSurfPointer;
                    laserCloudCubeCornerPointer->clear();
                    laserCloudCubeSurfPointer->clear();
                }
            }

            centerCubeI--;
            laserCloudCenWidth--;
        }

        while (centerCubeJ < 3)
        {
            for (int i = 0; i < laserCloudWidth; i++)
            {
                for (int k = 0; k < laserCloudDepth; k++)
                {
                    int j = laserCloudHeight - 1;
                    pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    for (; j >= 1; j--)
                    {
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudCornerArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudSurfArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
                    }
                    laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        laserCloudCubeCornerPointer;
                    laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        laserCloudCubeSurfPointer;
                    laserCloudCubeCornerPointer->clear();
                    laserCloudCubeSurfPointer->clear();
                }
            }

            centerCubeJ++;
            laserCloudCenHeight++;
        }

        while (centerCubeJ >= laserCloudHeight - 3)
        {
            for (int i = 0; i < laserCloudWidth; i++)
            {
                for (int k = 0; k < laserCloudDepth; k++)
                {
                    int j = 0;
                    pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    for (; j < laserCloudHeight - 1; j++)
                    {
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudCornerArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudSurfArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
                    }
                    laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        laserCloudCubeCornerPointer;
                    laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        laserCloudCubeSurfPointer;
                    laserCloudCubeCornerPointer->clear();
                    laserCloudCubeSurfPointer->clear();
                }
            }

            centerCubeJ--;
            laserCloudCenHeight--;
        }

        while (centerCubeK < 3)
        {
            for (int i = 0; i < laserCloudWidth; i++)
            {
                for (int j = 0; j < laserCloudHeight; j++)
                {
                    int k = laserCloudDepth - 1;
                    pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    for (; k >= 1; k--)
                    {
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
                    }
                    laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        laserCloudCubeCornerPointer;
                    laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        laserCloudCubeSurfPointer;
                    laserCloudCubeCornerPointer->clear();
                    laserCloudCubeSurfPointer->clear();
                }
            }

            centerCubeK++;
            laserCloudCenDepth++;
        }

        while (centerCubeK >= laserCloudDepth - 3)
        {
            for (int i = 0; i < laserCloudWidth; i++)
            {
                for (int j = 0; j < laserCloudHeight; j++)
                {
                    int k = 0;
                    pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    for (; k < laserCloudDepth - 1; k++)
                    {
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
                    }
                    laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        laserCloudCubeCornerPointer;
                    laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        laserCloudCubeSurfPointer;
                    laserCloudCubeCornerPointer->clear();
                    laserCloudCubeSurfPointer->clear();
                }
            }

            centerCubeK--;
            laserCloudCenDepth--;
        }
        // 至此，已能保证激光雷达在大cube内，且邻域可取值

        // 处理完毕边沿点，接下来就是在取到的子cube的5*5*5的邻域内找对应的配准点了。
        int laserCloudValidNum = 0;
        int laserCloudSurroundNum = 0;

        //5*5*3的邻域里进行循环寻找
        //在每一维附近5个cube(前2个，后2个，中间1个，k折半)里进行查找（每个cube边长为50m，前后共250米范围内），三个维度总共125个cube
        //在这125个cube里面进一步筛选在视域范围内的cube
        for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++) // x轴方向
        {
            for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++) // y轴方向
            {
                for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++) // k在3个cube中选 // z轴方向
                {
                    if (i >= 0 && i < laserCloudWidth && // 不超过大cube的边长
                        j >= 0 && j < laserCloudHeight &&
                        k >= 0 && k < laserCloudDepth)
                    { 
                        // 记录有效的特征点（不超出cube范围的），将领域5*5*3的小cube按次序放到数组laserCloudValidInd内
                        laserCloudValidInd[laserCloudValidNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                        laserCloudValidNum++;
                        // 记录了激光雷达附近的map（5X5X3）,用于发布
                        laserCloudSurroundInd[laserCloudSurroundNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                        laserCloudSurroundNum++;
                    }
                }
            }
        }

        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();
        // 将领域5*5*3的小cube中的角点和平面点分别按次序拼凑成laserCloudCornerFromMap和laserCloudSurfFromMap（用于被匹配）
        for (int i = 0; i < laserCloudValidNum; i++)
        {
            *laserCloudCornerFromMap += *laserCloudCornerArray[laserCloudValidInd[i]];
            *laserCloudSurfFromMap += *laserCloudSurfArray[laserCloudValidInd[i]];
        }
        
        // 进行两次下采样
        // 最后的输出结果是laserCloudCornerFromMapDS和laserCloudSurfFromMapDS
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->points.size();

        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->points.size();

        // cout<<"laserCloudCornerFromMapDSNum: "<<laserCloudCornerFromMapDSNum<<endl;
    }

    void downsampleCurrentScan(){

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

    bool InitICP(bool run)
    {
        // if (cloudKeyPoses3D->points.empty() == true)
        //     return true;
        if(run)
        {
            // downSizeFilterSurf.setInputCloud(laserCloudFullLast);
            // downSizeFilterSurf.filter(*laserCloudFullLastDS);

            // transform the lidar point cloud to map
            PointTypePose thisPose6D;
            thisPose6D.x = transformAftMapped[3];
            thisPose6D.y = transformAftMapped[4];
            thisPose6D.z = transformAftMapped[5];
            thisPose6D.roll = transformAftMapped[0];
            thisPose6D.pitch = transformAftMapped[1];
            thisPose6D.yaw = transformAftMapped[2];
            *laserCloudFullLastDS = *transformPointCloud(laserCloudFullLast, &thisPose6D);

            // copy the local map 
            *InitPointCloudMap += *laserCloudCornerFromMapDS;
            *InitPointCloudMap += *laserCloudSurfFromMapDS;

            // creat a icp tool
            pcl::IterativeClosestPoint<PointType, PointType> icp;
            icp.setMaxCorrespondenceDistance(50);
            icp.setMaximumIterations(100);
            icp.setTransformationEpsilon(1e-6);
            icp.setEuclideanFitnessEpsilon(1e-6);
            icp.setRANSACIterations(0);
            icp.setInputSource(laserCloudFullLastDS);
            icp.setInputTarget(InitPointCloudMap);
            pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
            icp.align(*unused_result); 
            cout<<"Initial ICP score: "<<icp.getFitnessScore()<<endl;
            float x_, y_, z_, roll_, pitch_, yaw_;
            Eigen::Affine3f correctionCameraFrame;
            correctionCameraFrame = icp.getFinalTransformation(); // get transformation in camera frame (because points are in camera frame)
            pcl::getTranslationAndEulerAngles(correctionCameraFrame, x_, y_, z_, roll_, pitch_, yaw_);
            Eigen::Affine3f correctionLidarFrame = pcl::getTransformation(z_, x_, y_, yaw_, roll_, pitch_);
            // transform from world origin to wrong pose
            Eigen::Affine3f tWrong = pclPointToAffine3fCameraToLidar(thisPose6D);
            // transform from world origin to corrected pose
            Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong; // pre-multiplying -> successive rotation about a fixed frame
            cout<<"correctionLidarFrame : \n"<<correctionCameraFrame.matrix()<<endl;
            pcl::getTranslationAndEulerAngles (tCorrect, z_, x_, y_, yaw_, roll_, pitch_);

            // //初始化正态分布变换（NDT）
            // pcl::NormalDistributionsTransform<PointType, PointType> ndt;
            // //设置依赖尺度NDT参数
            // //为终止条件设置最小转换差异
            // ndt.setTransformationEpsilon(0.01);
            // //为More-Thuente线搜索设置最大步长
            // ndt.setStepSize(0.5);
            // //设置NDT网格结构的分辨率（VoxelGridCovariance）
            // ndt.setResolution(1.0);
            // //设置匹配迭代的最大次数
            // ndt.setMaximumIterations(35);
            // // 设置要配准的点云
            // ndt.setInputCloud(laserCloudFullLastDS);
            // //设置点云配准目标
            // ndt.setInputTarget(InitPointCloudMap);
            // //计算需要的刚体变换以便将输入的点云匹配到目标点云
            // pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
            // ndt.align(*unused_result);
            // cout<<"Initial NDT score: "<<ndt.getFitnessScore()<<endl;
            // float x_, y_, z_, roll_, pitch_, yaw_;
            // Eigen::Affine3f correctionCameraFrame;
            // correctionCameraFrame = ndt.getFinalTransformation(); // get transformation in camera frame (because points are in camera frame)
            // pcl::getTranslationAndEulerAngles(correctionCameraFrame, x_, y_, z_, roll_, pitch_, yaw_);
            // Eigen::Affine3f correctionLidarFrame = pcl::getTransformation(z_, x_, y_, yaw_, roll_, pitch_);
            // // transform from world origin to wrong pose
            // Eigen::Affine3f tWrong = pclPointToAffine3fCameraToLidar(thisPose6D);
            // // transform from world origin to corrected pose
            // Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong; // pre-multiplying -> successive rotation about a fixed frame
            // cout<<"correctionLidarFrame : \n"<<correctionCameraFrame.matrix()<<endl;
            // pcl::getTranslationAndEulerAngles (tCorrect, z_, x_, y_, yaw_, roll_, pitch_);

            // update the transformTobeMapped and the transformAftMapped
            transformAftMapped[3] = x_;
            transformAftMapped[4] = y_;
            transformAftMapped[5] = z_;
            transformAftMapped[0] = roll_;
            transformAftMapped[1] = pitch_;
            transformAftMapped[2] = yaw_;

            for (int i = 0; i < 6; ++i){
            	transformTobeMapped[i] = transformAftMapped[i];
            }

            // thisPose6D.x = transformTobeMapped[3];
            // thisPose6D.y = transformTobeMapped[4];
            // thisPose6D.z = transformTobeMapped[5];
            // thisPose6D.roll = transformTobeMapped[0];
            // thisPose6D.pitch = transformTobeMapped[1];
            // thisPose6D.yaw = transformTobeMapped[2];

            // *laserCloudFullLastDS = *transformPointCloud(laserCloudFullLast, &thisPose6D);

            // sensor_msgs::PointCloud2 cloudMsgTemp;
            // pcl::toROSMsg(*laserCloudFullLastDS, cloudMsgTemp);
            // cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            // cloudMsgTemp.header.frame_id = "camera_init";
            // pubLaserCloudFullLast.publish(cloudMsgTemp);

            laserCloudCornerFromMap->clear();
            laserCloudCornerFromMapDS->clear();
            laserCloudCornerFromMapDSNum = 0;

            laserCloudSurfFromMap->clear();
            laserCloudSurfFromMapDS->clear();
            laserCloudSurfFromMapDSNum = 0;

            // set_init_guess = false;
            init_ICP = true;
            InitPointCloudMap->clear();

            return true;
        }
        else
        {
            return true;
        }
        
    }

    void cornerOptimization(int iterCount){

        updatePointAssociateToMapSinCos();
        for (int i = 0; i < laserCloudCornerLastDSNum; i++) {
            pointOri = laserCloudCornerLastDS->points[i];
            
            // 进行坐标变换,转换到全局坐标中去（世界坐标系）
            // pointSel:表示选中的点，point select
            // 输入是pointOri，输出是pointSel
            pointAssociateToMap(&pointOri, &pointSel);

            // 进行5邻域搜索，
            // pointSel为需要搜索的点，
            // pointSearchInd搜索完的邻域对应的索引
            // pointSearchSqDis 邻域点与查询点之间的距离
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
            
            // 只有当最远的那个邻域点的距离pointSearchSqDis[4]小于1m时才进行下面的计算
            // 以下部分的计算是在计算点集的协方差矩阵，Zhang Ji的论文中有提到这部分
            if (pointSearchSqDis[4] < 1.0) {
                // 先求5个样本的平均值
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                // 下面在求矩阵matA1=[ax,ay,az]^t*[ax,ay,az]
                // 更准确地说应该是在求协方差matA1
                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    // ax代表的是x-cx,表示均值与每个实际值的差值，求取5个之后再次取平均，得到matA1
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                // 求正交阵的特征值和特征向量
                // 特征值：matD1，特征向量：matV1中
                cv::eigen(matA1, matD1, matV1);

                // 边缘：与较大特征值相对应的特征向量代表边缘线的方向（一大两小，大方向）
                // 以下这一大块是在计算点到边缘的距离，最后通过系数s来判断是否距离很近
                // 如果距离很近就认为这个点在边缘上，需要放到laserCloudOri中
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

                    // 这边是在求[(x0-x1),(y0-y1),(z0-z1)]与[(x0-x2),(y0-y2),(z0-z2)]叉乘得到的向量的模长
                    // 这个模长是由0.2*V1[0]和点[x0,y0,z0]构成的平行四边形的面积
                    // 因为[(x0-x1),(y0-y1),(z0-z1)]x[(x0-x2),(y0-y2),(z0-z2)]=[XXX,YYY,ZZZ],
                    // [XXX,YYY,ZZZ]=[(y0-y1)(z0-z2)-(y0-y2)(z0-z1),-(x0-x1)(z0-z2)+(x0-x2)(z0-z1),(x0-x1)(y0-y2)-(x0-x2)(y0-y1)]
                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                    * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                    * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))
                                    * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    // l12表示的是0.2*(||V1[0]||)
                    // 也就是平行四边形一条底的长度
                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    // 求叉乘结果[la',lb',lc']=[(x1-x2),(y1-y2),(z1-z2)]x[XXX,YYY,ZZZ]
                    // [la,lb,lc]=[la',lb',lc']/a012/l12
                    // LLL=[la,lb,lc]是0.2*V1[0]这条高上的单位法向量。||LLL||=1；
                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;
                    
                    // 计算点pointSel到直线的距离
                    // 这里需要特别说明的是ld2代表的是点pointSel到过点[cx,cy,cz]的方向向量直线的距离
                    float ld2 = a012 / l12;

                    // 如果在最理想的状态的话，ld2应该为0，表示点在直线上
                    // 最理想状态s=1；
                    float s = 1 - 0.9 * fabs(ld2);
                    
                    // coeff代表系数的意思
                    // coff用于保存距离的方向向量
                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;

                    // intensity本质上构成了一个核函数，ld2越接近于1，增长越慢
                    // intensity=(1-0.9*ld2)*ld2=ld2-0.9*ld2*ld2
                    coeff.intensity = s * ld2;
                    
                    // 所以就应该认为这个点是边缘点
                    // s>0.1 也就是要求点到直线的距离ld2要小于1m
                    // s越大说明ld2越小(离边缘线越近)，这样就说明点pointOri在直线上
                    if (s > 0.1) {
                        laserCloudOri->push_back(pointOri);
                        coeffSel->push_back(coeff);
                    }
                }
            }
        }
    }

    void surfOptimization(int iterCount){
        updatePointAssociateToMapSinCos();
        for (int i = 0; i < laserCloudSurfTotalLastDSNum; i++) {
            pointOri = laserCloudSurfTotalLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel); 
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0.at<float>(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0.at<float>(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0.at<float>(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                // matB0是一个5x1的矩阵
                // matB0 = cv::Mat (5, 1, CV_32F, cv::Scalar::all(-1));
                // matX0是3x1的矩阵
                // 求解方程matA0*matX0=matB0
                // 公式其实是在求由matA0中的点构成的平面的法向量matX0
                cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);

                // [pa,pb,pc,pd]=[matX0,pd]
                // 正常情况下（见后面planeValid判断条件），应该是
                // pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                // pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                // pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z = -1
                // 所以pd设置为1
                float pa = matX0.at<float>(0, 0);
                float pb = matX0.at<float>(1, 0);
                float pc = matX0.at<float>(2, 0);
                float pd = 1;

                // 对[pa,pb,pc,pd]进行单位化
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                // 求解后再次检查平面是否是有效平面
                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    // 后面部分相除求的是[pa,pb,pc,pd]与pointSel的夹角余弦值(两个sqrt，其实并不是余弦值)
                    // 这个夹角余弦值越小越好，越小证明所求的[pa,pb,pc,pd]与平面越垂直
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    // 判断是否是合格平面，是就加入laserCloudOri
                    if (s > 0.1) {
                        laserCloudOri->push_back(pointOri);
                        coeffSel->push_back(coeff);
                    }
                }
            }
        }
    }

    // 这部分的代码是基于高斯牛顿法的优化，不是zhang ji论文中提到的基于L-M的优化方法
    // 这部分的代码使用旋转矩阵对欧拉角求导，优化欧拉角，不是zhang ji论文中提到的使用angle-axis的优化
    bool LMOptimization(int iterCount){
        float srx = sin(transformTobeMapped[0]);
        float crx = cos(transformTobeMapped[0]);
        float sry = sin(transformTobeMapped[1]);
        float cry = cos(transformTobeMapped[1]);
        float srz = sin(transformTobeMapped[2]);
        float crz = cos(transformTobeMapped[2]);

        int laserCloudSelNum = laserCloudOri->points.size();
        // laser cloud original 点云太少，就跳过这次循环
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

            // 求雅克比矩阵中的元素，距离d对roll角度的偏导量即d(d)/d(roll)
            // 更详细的数学推导参看wykxwyc.github.io
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            // 同上，求解的是对pitch的偏导量
            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;


            /*
            在求点到直线的距离时，coeff表示的是如下内容
            [la,lb,lc]表示的是点到直线的垂直连线方向，s是长度
            coeff.x = s * la;
            coeff.y = s * lb;
            coeff.z = s * lc;
            coeff.intensity = s * ld2;

            在求点到平面的距离时，coeff表示的是
            [pa,pb,pc]表示过外点的平面的法向量，s是线的长度
            coeff.x = s * pa;
            coeff.y = s * pb;
            coeff.z = s * pc;
            coeff.intensity = s * pd2;
            */
            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;

            // 这部分是雅克比矩阵中距离对平移的偏导
            matA.at<float>(i, 3) = coeff.x;
            matA.at<float>(i, 4) = coeff.y;
            matA.at<float>(i, 5) = coeff.z;

            // 残差项
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        // 将矩阵由matA转置生成matAt
        // 先进行计算，以便于后边调用 cv::solve求解
        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;

        // 利用高斯牛顿法进行求解，
        // 高斯牛顿法的原型是J^(T)*J * delta(x) = -J*f(x)
        // J是雅克比矩阵，这里是A，f(x)是优化目标，这里是-B(符号在给B赋值时候就放进去了)
        // 通过QR分解的方式，求解matAtA*matX=matAtB，得到解matX
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        // iterCount==0 说明是第一次迭代，需要初始化
        if (iterCount == 0) {
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            // 对近似的Hessian矩阵求特征值和特征向量，
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

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        // 旋转或者平移量足够小就停止这次迭代过程
        if (deltaR < 0.05 && deltaT < 0.05) {
            return true;
        }
        return false;
    }

    void scan2MapOptimization(){
        // laserCloudCornerFromMapDSNum是extractSurroundingKeyFrames()函数最后降采样得到的coner点云数
        // laserCloudSurfFromMapDSNum是extractSurroundingKeyFrames()函数降采样得到的surface点云数
        // 附近点云地图点的数目是否足够
        if (laserCloudCornerFromMapDSNum > 10 && laserCloudSurfFromMapDSNum > 100) {

            // laserCloudCornerFromMapDS和laserCloudSurfFromMapDS的来源有2个：
            // 当有闭环时，来源是：recentCornerCloudKeyFrames，没有闭环时，来源surroundingCornerCloudKeyFrames
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            for (int iterCount = 0; iterCount < 10; iterCount++) {
                // 用for循环控制迭代次数，最多迭代10次
                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization(iterCount);
                surfOptimization(iterCount);

                if (LMOptimization(iterCount) == true)
                    break;              
            }

            // 迭代结束更新相关的转移矩阵
            transformUpdate();
        }
    }


    void saveKeyFramesAndFactor(){

        currentRobotPosPoint.x = transformAftMapped[3];
        currentRobotPosPoint.y = transformAftMapped[4];
        currentRobotPosPoint.z = transformAftMapped[5];

        bool saveThisKeyFrame = true;
        // 由于localization不需要回环，因此这里不需要稀疏化
        // 当前帧与上一关键帧距离有0.3m以上才保存为关键帧
        // if (sqrt((previousRobotPosPoint.x-currentRobotPosPoint.x)*(previousRobotPosPoint.x-currentRobotPosPoint.x)
        //         +(previousRobotPosPoint.y-currentRobotPosPoint.y)*(previousRobotPosPoint.y-currentRobotPosPoint.y)
        //         +(previousRobotPosPoint.z-currentRobotPosPoint.z)*(previousRobotPosPoint.z-currentRobotPosPoint.z)) < 0.5){
        //     saveThisKeyFrame = false;
        // }

        if (saveThisKeyFrame == false && !cloudKeyPoses3D->points.empty())
        	return;

        previousRobotPosPoint = currentRobotPosPoint;

        ROS_DEBUG("cloudKeyPoses3D->points.size() = %d\n", cloudKeyPoses3D->points.size());

        // 如果当前帧为第一个关键帧,则需要初始化因子图
        if (cloudKeyPoses3D->points.empty()){
            // // static Rot3 	RzRyRx (double x, double y, double z),Rotations around Z, Y, then X axes
            // // RzRyRx依次按照z(transformTobeMapped[2])，y(transformTobeMapped[0])，x(transformTobeMapped[1])坐标轴旋转
            // // Point3 (double x, double y, double z)  Construct from x(transformTobeMapped[5]), y(transformTobeMapped[3]), and z(transformTobeMapped[4]) coordinates. 
            // // Pose3 (const Rot3 &R, const Point3 &t) Construct from R,t. 从旋转和平移构造姿态
            // // NonlinearFactorGraph增加一个PriorFactor因子
            // gtSAMgraph.add(PriorFactor<Pose3>(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
            //                                            		 Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])), priorNoise));
            // // initialEstimate的数据类型是Values,其实就是一个map，这里在0对应的值下面保存了一个Pose3
            // // 也就是插入该顶点的初值用于迭代
            // initialEstimate.insert(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
            //                                       Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])));
            // for (int i = 0; i < 6; ++i)
            // 	transformLast[i] = transformTobeMapped[i];

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
        }
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
        else if(!useBetweenFactor)
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
        if(!sleipnir_gps_queue.empty())
        {
            sleipnir_msgs::sensorgps lastGps;
            sleipnir_msgs::sensorgps currGps;
            // extract the current gps message
            while(!sleipnir_gps_queue.empty())
            {
                if(sleipnir_gps_queue.front().header.stamp.toSec()<timeLaserOdometry-0.1)
                {
                    lastGps = sleipnir_gps_queue.front();
                    sleipnir_gps_queue.pop();
                }
                else
                break;
            }
            if(!sleipnir_gps_queue.empty()) // there are proper data in the queue
            {
                if(sleipnir_gps_queue.front().header.stamp.toSec()>timeLaserOdometry-0.1 &&
                sleipnir_gps_queue.front().header.stamp.toSec()-timeLaserOdometry<0.1)
                {
                    currGps = sleipnir_gps_queue.front();
                    sleipnir_msgs::sensorgps newGps = GpsSlerp(lastGps, currGps, timeLaserOdometry);

                    // cout<<"hello !"<<endl;
                    // // 这里应该从文件中读取地图起点处的GPS数据，然后将其转换为UTM的起点，每个在线数据都减去该起点坐标
                    // if(!init_fix)
                    // {
                    //     LLtoUTM(OriLat, OriLon, init_fix_odom_y, init_fix_odom_x, init_fix_zoom);
                    //     init_fix_odom_z = OriAlt;

                    //     // 我们车上使用的星网宇达GPS接收器，他的航向角方向与笛卡尔直角坐标系相反
                    //     InitEulerAngle=Eigen::Vector3d((-OriYaw+90.0)*deg+compensate_init_yaw, -OriPitch*deg, OriRoll*deg);
                    //     rollAngle = (AngleAxisd(InitEulerAngle(2),Vector3d::UnitX()));
                    //     pitchAngle = (AngleAxisd(InitEulerAngle(1),Vector3d::UnitY()));
                    //     yawAngle = (AngleAxisd(InitEulerAngle(0),Vector3d::UnitZ()));
                    //     // init_fix_odom_pose = rollAngle*pitchAngle*yawAngle;
                    //     init_fix_odom_pose = pitchAngle * yawAngle * rollAngle;
                    //     init_fix_odom_pose = init_fix_odom_pose.inverse();
                    //     init_fix = true;
                    // }

                    double northing, easting;
                    std::string zone;

                    // LLtoUTM(newGps.lat, newGps.lon, northing, easting, zone);
                    bool northp;
                    int izone;
                    UTMUPS::Forward(newGps.lat, newGps.lon, izone, northp, easting, northing);
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
                    fix_odom_position = yawAngle.inverse() * fix_odom_position;

                    fix_odom.pose.pose.position.x = fix_odom_position.y();
                    fix_odom.pose.pose.position.y = fix_odom_position.z();
                    fix_odom.pose.pose.position.z = fix_odom_position.x();

                    // cout<<"GPS YAW："<<-newGps.heading+90.0+compensate_init_yaw*rad-InitEulerAngle(0)*rad<<endl;
                    // cout<<"Fusion YAW："<<transformTobeMapped[1]*rad<<endl;
                    
                    Eigen::Vector3d CurreulerAngle((-newGps.heading+90.0)*deg+compensate_init_yaw, newGps.roll*deg, -newGps.pitch*deg);
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
                    GPSHistoryPosition3D->push_back(gps_position);
                    sensor_msgs::PointCloud2 gps_position_cloudMsgTemp;
                    pcl::toROSMsg(*GPSHistoryPosition3D, gps_position_cloudMsgTemp);
                    gps_position_cloudMsgTemp.header.stamp = newGps.header.stamp;
                    gps_position_cloudMsgTemp.header.frame_id = "camera_init";
                    fix_position_pub.publish(gps_position_cloudMsgTemp);

                    // mark the unreliable point for visualization
                    // cout<<"status value: "<<newGps.status<<endl;
                    // 0：初始化 48
                    // 1：粗对准 
                    // 2：精对准
                    // 3：GPS定位
                    // 4：GPS定向
                    // 5：RTK    53
                    // 6：DMI组合
                    // 7：DMI标定
                    // 8：纯惯性
                    // 9：零速校正 57
                    // A：VG模式 65
                    // B：差分定向 66
                    // C：动态对准 67
                    Eigen::Vector3d diffBetLidarAndGps;
                    diffBetLidarAndGps.x() = transformAftMapped[5] - gps_position.z;
                    diffBetLidarAndGps.y() = transformAftMapped[3] - gps_position.x;
                    diffBetLidarAndGps.z() = transformAftMapped[4] - gps_position.y;
                    double disGps2Lidar = diffBetLidarAndGps.norm();
                    // cout<<"disGps2Lidar: "<<disGps2Lidar<<endl;

                    switch(newGps.status)
                    {
                        case 'B':
                        {
                            gps_noise_x = 1.0;
                            gps_noise_y = 1.0;
                            gps_noise_z = 10.0;

                            angle_noise = 0.01;
                            indoor = false;
                            // cout<<"status is good!"<<endl;
                            break;
                        }
                        case '5':
                        {
                            gps_noise_x = 1.0;
                            gps_noise_y = 1.0;
                            gps_noise_z = 10.0;
                            angle_noise = 0.01;
                            indoor = false;
                            // cout<<"status is not so good!"<<endl;
                            break;
                        }
                        case '4':
                        {
                            gps_noise_x = 60.0;
                            gps_noise_y = 60.0;
                            gps_noise_z = 100.0;

                            angle_noise = 0.1;
                            indoor = false;
                            // cout<<"status is not so good!"<<endl;
                            break;
                        }
                        case '3':
                        {
                            gps_noise_x = 80.0;
                            gps_noise_y = 80.0;
                            gps_noise_z = 100.0;

                            angle_noise = 0.1;
                            indoor = false;
                            // cout<<"status is not so good!"<<endl;
                            break;
                        }
                        default:
                        {
                            indoor = true;
                            buffNum++;
                            if(buffNum > 200) buffNum = 200;

                            // gps_noise_x = 80.0;
                            // gps_noise_y = 80.0;
                            // gps_noise_z = 80.0;

                            // angle_noise = 0.1;

                            // cout<<"status is bad!"<<endl;
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
                        indoor = true;
                    }
                    // cout<<"gps_noise_x: "<<gps_noise_x<<endl;
                    
                    bool add_gps_only = false;
                    // if(add_gps_only && !indoor)
                    // {
                    //     gtsam::noiseModel::Diagonal::shared_ptr GPSNoise = 
                    //     noiseModel::Diagonal::Sigmas((gtsam::Vector(3) <<gps_noise_x, gps_noise_y, gps_noise_z).finished());
                    //     gtsam::GPSFactor gps_factor(cloudKeyPoses3D->points.size(), 
                    //                                 gtsam::Point3(gps_position.z, gps_position.x, gps_position.y), 
                    //                                 GPSNoise);
                    // }

                    // if(!add_gps_only && !indoor)
                    if(!add_gps_only && !indoor && useGPSfactor)
                    {
                        gtsam::noiseModel::Diagonal::shared_ptr GPS_Pose3Noise = 
                        noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << angle_noise, angle_noise, angle_noise, gps_noise_x, gps_noise_y, gps_noise_z).finished());
                        gtsam::PriorFactor<gtsam::Pose3> GPS_Pose3Factor(X(cloudKeyPoses3D->points.size()), 
                                                                        gtsam::Pose3(gtsam::Rot3::Quaternion(fix_odom.pose.pose.orientation.w,
                                                                                                        fix_odom.pose.pose.orientation.z,
                                                                                                        fix_odom.pose.pose.orientation.x,
                                                                                                        fix_odom.pose.pose.orientation.y),
                                                                                    gtsam::Point3(gps_position.z, gps_position.x, gps_position.y)), 
                                                                                    GPS_Pose3Noise);
                        gtSAMgraph.add(GPS_Pose3Factor);
                        // cout<<"add GPS!~~"<<cloudKeyPoses3D->points.size()<<endl;
                    }

                    sleipnir_gps_queue.pop();
                }
            }
        }

        // gtsam::ISAM2::update函数原型:
        // ISAM2Result gtsam::ISAM2::update	(	const NonlinearFactorGraph & 	newFactors = NonlinearFactorGraph(),
        // const Values & 	newTheta = Values(),
        // const std::vector< size_t > & 	removeFactorIndices = std::vector<size_t>(),
        // const boost::optional< FastMap< Key, int > > & 	constrainedKeys = boost::none,
        // const boost::optional< FastList< Key > > & 	noRelinKeys = boost::none,
        // const boost::optional< FastList< Key > > & 	extraReelimKeys = boost::none,
        // bool 	force_relinearize = false )	
        // gtSAMgraph是新加到系统中的因子
        // initialEstimate是加到系统中的新变量的初始点
        isam->update(gtSAMgraph, initialEstimate);
        // update 函数为什么需要调用两次？
        isam->update();

		// 删除内容?
        gtSAMgraph.resize(0);
		initialEstimate.clear();

        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        // Compute an estimate from the incomplete linear delta computed during the last update.
        isamCurrentEstimate = isam->calculateEstimate();
        // latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        latestEstimate = isamCurrentEstimate.at<Pose3>(X(cloudKeyPoses3D->points.size()));

        thisPose3D.x = latestEstimate.translation().y();
        thisPose3D.y = latestEstimate.translation().z();
        thisPose3D.z = latestEstimate.translation().x();
        thisPose3D.intensity = cloudKeyPoses3D->points.size();
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity;
        thisPose6D.roll  = latestEstimate.rotation().pitch();
        thisPose6D.pitch = latestEstimate.rotation().yaw();
        thisPose6D.yaw   = latestEstimate.rotation().roll();
        thisPose6D.time = timeLaserOdometry;
        cloudKeyPoses6D->push_back(thisPose6D);
//        cout<<"thisPose6D translation:    x="<<thisPose6D.x<<", y="<<thisPose6D.y<<", z="<<thisPose6D.z<<endl;
//        cout<<"thisPose6D orientation: yaw="<<thisPose6D.yaw/PI*180<<", pitch="<<thisPose6D.pitch/PI*180<<", roll="<<thisPose6D.roll/PI*180<<endl<<endl;

        if (cloudKeyPoses3D->points.size() > 1){
            transformAftMapped[0] = latestEstimate.rotation().pitch();
            transformAftMapped[1] = latestEstimate.rotation().yaw();
            transformAftMapped[2] = latestEstimate.rotation().roll();
            transformAftMapped[3] = latestEstimate.translation().y();
            transformAftMapped[4] = latestEstimate.translation().z();
            transformAftMapped[5] = latestEstimate.translation().x();

            for (int i = 0; i < 6; ++i){
            	transformLast[i] = transformAftMapped[i];
            	transformTobeMapped[i] = transformAftMapped[i];
            }
        }
    }

    void clearCloud(){
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();
        laserCloudCornerFromMapDS->clear();
        laserCloudSurfFromMapDS->clear();   
    }

    void run(){

        tf::StampedTransform test_transform;
        try{
            pclmap_2_planning_odom_listener.waitForTransform("pcl_map", rviz_frame, ros::Time().fromSec(timeLaserOdometry), ros::Duration(3.0));
            pclmap_2_planning_odom_listener.lookupTransform("pcl_map", rviz_frame,
                               ros::Time().fromSec(timeLaserOdometry), test_transform);
        }
        catch(tf::TransformException& ex){
            // ROS_ERROR("Received an exception trying to transform a point from \"camera\" to \"pcl_map\": %s", ex.what());
            ROS_ERROR("%s",ex.what());
            ros::Duration(4.0).sleep();
        }
        // cout<<"test_transform: "<<test_transform.getOrigin().x()<<endl;

        // 判断是否有新的数据到来并且时间差值小于0.005
        if (newLaserCloudCornerLast  && std::abs(timeLaserCloudCornerLast  - timeLaserOdometry) < 0.005 &&
            newLaserCloudSurfLast    && std::abs(timeLaserCloudSurfLast    - timeLaserOdometry) < 0.005 &&
            newLaserCloudOutlierLast && std::abs(timeLaserCloudOutlierLast - timeLaserOdometry) < 0.005 &&
            newLaserCloudFullLast    && std::abs(timeLaserCloudFullLast    - timeLaserOdometry) < 0.005 &&
            newLaserOdometry &&
            set_init_guess)
        {

            // 标志位复位
            newLaserCloudCornerLast = false; newLaserCloudSurfLast = false; newLaserCloudOutlierLast = false; newLaserOdometry = false;
            newLaserCloudFullLast = false;

            std::lock_guard<std::mutex> lock(mtx);

            //mappingProcessInterval是0.3秒，以相对较慢的速度进行建图
            if (timeLaserOdometry - timeLastProcessing >= mappingProcessInterval ) {

                timeLastProcessing = timeLaserOdometry;

                //把点云坐标均转换到世界坐标系下, 更新transformTobeMapped = transformAftMapped + (transformSum - transformBefMapped)
                transformAssociateToMap();

                //由于帧数的频率大于建图的频率，因此需要提取关键帧进行匹配
                // 实际是构建局部地图,需要大改
                extractSurroundingKeyFrames();
                
                // 插入初始化函数，将原始的激光雷达与局部地图进行ICP,只运行一次
                // if(!InitICP(set_init_guess)) return;
                InitICP(!init_ICP);

                //对当前雷达点云帧进行降采样
                downsampleCurrentScan();

                // ICP优化
                scan2MapOptimization();

                // 留着以后说不定可以跟GPS融合
                saveKeyFramesAndFactor();

                // correctPoses();

                publishTF();

                publishKeyPosesAndFrames();

                clearCloud();
            }
        }
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lego_loam");

    ROS_INFO("\033[1;32m---->\033[0m Map Optimization Started.");

    ros::NodeHandle pnh("~");
    parameter::readParameters(pnh);
    parameter::readInitPose(pnh);

    mapOptimization MO;

    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::Rate rate(200);
    while (ros::ok())
    {
        ros::spinOnce();

        MO.run();

        rate.sleep();
    }

    visualizeMapThread.join();

    return 0;
}



