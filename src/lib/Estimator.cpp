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

#include <Estimator.h>

namespace fusion {

int Scan::scan_counter_ = 0;

ofstream fout_evo;
// clock_t time_stt;

LinsFusion::LinsFusion(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    : nh_(nh), pnh_(pnh) {}

LinsFusion::~LinsFusion() { delete estimator; }

void LinsFusion::run() { initialization(); }

void LinsFusion::initialization() {
  // Implement an iterative-ESKF Kalman filter class
  // 新建一个状态估计器
  estimator = new StateEstimator();

  // Subscribe to IMU, segmented point clouds, and map-refined odometry feedback
  // 经map修正的里程计反馈
  subMapOdom_ = pnh_.subscribe<nav_msgs::Odometry>(
      LIDAR_MAPPING_TOPIC, 5, &LinsFusion::mapOdometryCallback, this);
  // IMU数据
  subImu = pnh_.subscribe<sensor_msgs::Imu>(IMU_TOPIC, 100,
                                            &LinsFusion::imuCallback, this);
  // 预处理过的点云，它们的回调函数只是把点云数据压入队列
  subLaserCloud = pnh_.subscribe<sensor_msgs::PointCloud2>(
      "/segmented_cloud", 2, &LinsFusion::laserCloudCallback, this);
  subLaserCloudInfo = pnh_.subscribe<cloud_msgs::cloud_info>(
      "/segmented_cloud_info", 2, &LinsFusion::laserCloudInfoCallback, this);
  subOutlierCloud = pnh_.subscribe<sensor_msgs::PointCloud2>(
      "/outlier_cloud", 2, &LinsFusion::outlierCloudCallback, this);

  // Set publishers
  pubUndistortedPointCloud =
      pnh_.advertise<sensor_msgs::PointCloud2>("/undistorted_point_cloud", 1);

  // 发布特征点云
  pubCornerPointsSharp =
      pnh_.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 1);
  pubCornerPointsLessSharp =
      pnh_.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 1);
  pubSurfPointsFlat =
      pnh_.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 1);
  pubSurfPointsLessFlat =
      pnh_.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 1);

  pubLaserCloudCornerLast =
      pnh_.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2);
  pubLaserCloudSurfLast =
      pnh_.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2);
  pubOutlierCloudLast =
      pnh_.advertise<sensor_msgs::PointCloud2>("/outlier_cloud_last", 2);
  pubLaserOdometry =
      pnh_.advertise<nav_msgs::Odometry>(LIDAR_ODOMETRY_TOPIC, 5);

  // Set types of the point cloud
  distortedPointCloud.reset(new pcl::PointCloud<PointType>());
  outlierPointCloud.reset(new pcl::PointCloud<PointType>());

  // Allocate measurement buffers for sensors
  imuBuf_.allocate(500);
  pclBuf_.allocate(3);
  outlierBuf_.allocate(3);
  cloudInfoBuf_.allocate(3);

  // Initialize IMU propagation parameters
  isImuCalibrated = CALIBARTE_IMU;
  ba_init_ = INIT_BA;
  bw_init_ = INIT_BW;
  ba_tmp_.setZero();
  bw_tmp_.setZero();
  sample_counter_ = 0;

  duration_ = 0.0;
  scan_counter_ = 0;

  ROS_INFO_STREAM("Subscribe to \033[1;32m---->\033[0m " << IMU_TOPIC);
  ROS_INFO_STREAM("Subscribe to \033[1;32m---->\033[0m " << LIDAR_TOPIC);

  fout_evo.open("/home/kyle/Downloads/ROSfile/lins_odom_traj_evo.txt");
  fout_evo.precision(15);
  fout_evo.clear();

  // time_stt = clock();
}

void LinsFusion::laserCloudCallback(
    const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg) {
  // Add a new segmented point cloud
  pclBuf_.addMeas(laserCloudMsg, laserCloudMsg->header.stamp.toSec());
}
void LinsFusion::laserCloudInfoCallback(
    const cloud_msgs::cloud_infoConstPtr& cloudInfoMsg) {
  // Add segmentation information of the point cloud
  cloudInfoBuf_.addMeas(*cloudInfoMsg, cloudInfoMsg->header.stamp.toSec());
}

void LinsFusion::outlierCloudCallback(
    const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg) {
  outlierBuf_.addMeas(laserCloudMsg, laserCloudMsg->header.stamp.toSec());
}

// 获取由tansform_fusion_node发布的经map修正的odometry数据
void LinsFusion::mapOdometryCallback(
    const nav_msgs::Odometry::ConstPtr& odometryMsg) {
  // 获取当前车体在map中的姿态
  geometry_msgs::Quaternion geoQuat = odometryMsg->pose.pose.orientation;
  // 位置
  V3D t_yzx(odometryMsg->pose.pose.position.x,
            odometryMsg->pose.pose.position.y,
            odometryMsg->pose.pose.position.z);
  // geometry_msgs::Quaternion -> Eigen::Quaterniond
  Q4D q_yzx(geoQuat.w, geoQuat.x, geoQuat.y, geoQuat.z);
  // 将YZX约定旋转顺序下的transition转换到XYZ约定旋转顺序下
  V3D t_xyz = estimator->Q_yzx_to_xyz * t_yzx;
  // 将YZX约定旋转顺序下的rotation转换到XYZ约定旋转顺序下
  Q4D q_xyz =
      estimator->Q_yzx_to_xyz * q_yzx * estimator->Q_yzx_to_xyz.inverse();

  // quaternion --> rpy
  V3D rpy = math_utils::Q2rpy(q_xyz);
}

void LinsFusion::imuCallback(const sensor_msgs::Imu::ConstPtr& imuMsg) {
  // Align IMU measurements from IMU frame to vehicle frame
  // two frames share same roll and pitch angles, but with a small
  // misalign-angle in the yaw direction
  // 三轴加速度原始数据
  acc_raw_ << imuMsg->linear_acceleration.x, imuMsg->linear_acceleration.y,
      imuMsg->linear_acceleration.z;
  // 三轴旋转速度原始数据
  gyr_raw_ << imuMsg->angular_velocity.x, imuMsg->angular_velocity.y,
      imuMsg->angular_velocity.z;
  // imu frame和vehicle fram在roll & pitch方向上相同，在yaw上有小误差(默认为3.0°)
  misalign_euler_angles_ << deg2rad(0.0), deg2rad(0.0),
      deg2rad(IMU_MISALIGN_ANGLE);
  // 在Imu数据的基础上添加上外参，转换到车体坐标系下
  alignIMUtoVehicle(misalign_euler_angles_, acc_raw_, gyr_raw_, acc_aligned_,
                    gyr_aligned_);

  // Add a new IMU measurement
  // 将该处理后的imu测量压入队列
  Imu imu(imuMsg->header.stamp.toSec(), acc_aligned_, gyr_aligned_);
  imuBuf_.addMeas(imu, imuMsg->header.stamp.toSec());

  // Trigger the Kalman filter
  performStateEstimation();
}

void LinsFusion::processFirstPointCloud() {
  // Use the most recent point cloud to initialize the estimator
  // 队列中最新一帧点云数据的时间戳
  pclBuf_.getLastTime(scan_time_);

  // 问题来了,他怎么保证下面提取的三种点云帧的时间戳是对齐的???
  
  // 取出队列中最新一帧点云
  sensor_msgs::PointCloud2::ConstPtr pclMsg;
  pclBuf_.getLastMeas(pclMsg);
  // msg --> pcl
  distortedPointCloud->clear();
  pcl::fromROSMsg(*pclMsg, *distortedPointCloud);

  // 聚类数量不够30,列数为5的倍数，并且行数较大，可以认为非地面的点--> 局外点
  sensor_msgs::PointCloud2::ConstPtr outlierMsg;
  outlierBuf_.getLastMeas(outlierMsg);
  outlierPointCloud->clear();
  pcl::fromROSMsg(*outlierMsg, *outlierPointCloud);

  // 他的强度值存储着各点的ring数
  cloud_msgs::cloud_info cloudInfoMsg;
  cloudInfoBuf_.getLastMeas(cloudInfoMsg);

  // The latest IMU measurement records the inertial information when the new
  // point cloud is recorded
  // 该Imu测量记录了该帧点云对应的惯性测量数据
  Imu imu;
  // 提取最新一帧imu
  imuBuf_.getLastMeas(imu);

  // 提取的数据：scan_time_, distortedPointCloud, outlierPointCloud, cloudInfoMsg, imu

  // Initialize the iterative-ESKF by the first PCL
  // 使用I-ESKF进行紧耦合并更新状态
  estimator->processPCL(scan_time_, imu, distortedPointCloud, cloudInfoMsg,
                        outlierPointCloud);

  // Clear all the PCLs before the initialization PCL
  pclBuf_.clean(estimator->getTime());
  cloudInfoBuf_.clean(estimator->getTime());
  outlierBuf_.clean(estimator->getTime());
}

// 发布点云到topic
void LinsFusion::publishTopics() {
  if (pubLaserCloudCornerLast.getNumSubscribers() != 0) {
    publishCloudMsg(pubLaserCloudCornerLast,
                    estimator->scan_last_->cornerPointsLessSharpYZX_,
                    ros::Time().fromSec(scan_time_), "/loam_camera");
  }
  if (pubLaserCloudSurfLast.getNumSubscribers() != 0) {
    publishCloudMsg(pubLaserCloudSurfLast,
                    estimator->scan_last_->surfPointsLessFlatYZX_,
                    ros::Time().fromSec(scan_time_), "/loam_camera");
  }
  if (pubOutlierCloudLast.getNumSubscribers() != 0) {
    publishCloudMsg(pubOutlierCloudLast,
                    estimator->scan_last_->outlierPointCloudYZX_,
                    ros::Time().fromSec(scan_time_), "/loam_camera");
  }

  // Publish the estimated 6-DOF odometry by a YZX-frame convention (e.g. camera
  // frame convention), where Z points forward, X poins leftward, and Y poitns
  // upwards.
  // Notive that the estimator is performed in a XYZ-frame convention,
  // where X points forward, Y...leftward, Z...upward. Therefore, we have to
  // transforme the odometry from XYZ-convention to YZX-convention to meet the
  // mapping module's requirement.
  publishOdometryYZX(scan_time_);
}

bool LinsFusion::processPointClouds() {
  // Obtain the next PCL
  // upper_bound(num)从数组的begin位置到end-1位置二分查找第一个大于num的数字，找到返回该数字的地址，
  // 不存在则返回end。通过返回的地址减去起始地址begin,得到找到数字在数组中的下标。
  // 找到一帧时间戳在状态预测器当前时间戳之后的点云帧
  pclBuf_.itMeas_ = pclBuf_.measMap_.upper_bound(estimator->getTime());
  // 封装点云帧的消息指针
  sensor_msgs::PointCloud2::ConstPtr pclMsg = pclBuf_.itMeas_->second;
  // 点云帧的时间戳
  scan_time_ = pclBuf_.itMeas_->first;
  distortedPointCloud->clear();
  // 提取点云消息中的点云数据
  pcl::fromROSMsg(*pclMsg, *distortedPointCloud);

  // 同上，提取出局外点
  outlierBuf_.itMeas_ = outlierBuf_.measMap_.upper_bound(estimator->getTime());
  sensor_msgs::PointCloud2::ConstPtr outlierMsg = outlierBuf_.itMeas_->second;
  outlierPointCloud->clear();
  pcl::fromROSMsg(*outlierMsg, *outlierPointCloud);

  cloudInfoBuf_.itMeas_ =
      cloudInfoBuf_.measMap_.upper_bound(estimator->getTime());
  cloud_msgs::cloud_info cloudInfoMsg = cloudInfoBuf_.itMeas_->second;

  // 提取最新一帧imu的时间戳
  imuBuf_.getLastTime(last_imu_time_);
  // 最新一帧imu数据应该在点云帧以后的,这样才有充分足够的Imu数据进行预积分
  if (last_imu_time_ < scan_time_) {
    // ROS_WARN("Wait for more IMU measurement!");
    return false;
  }

  // Propagate IMU measurements between two consecutive scans
  int imu_couter = 0;
  // 预测器时间在点云帧之前 且 imu数据数据队列中存在预测器时间戳以后的数据
  //                 {<-----pre-intergration--->}
  // ----------------+--------------------------+-----------------+-----------> t
  //         first-estimate_time            scan_time       last_imu_time
  while (estimator->getTime() < scan_time_ &&
         (imuBuf_.itMeas_ = imuBuf_.measMap_.upper_bound(
              estimator->getTime())) != imuBuf_.measMap_.end()) {
    double dt =
        std::min(imuBuf_.itMeas_->first, scan_time_) - estimator->getTime();
    // 提取imu数据
    Imu imu = imuBuf_.itMeas_->second;
    // 注意这期间会一直通过dt更新estimator->getTime()的值
    estimator->processImu(dt, imu.acc, imu.gyr);
  }
  //                   (complete pre-intergration)
  // ---------------------------+--+-----------------+-----------> t
  //                           scan_time       last_imu_time
  //                      estimate_time

  Imu imu;
  // 保留最新一帧Imu数据
  imuBuf_.getLastMeas(imu);

  // Update the iterative-ESKF using a new PCL
  estimator->processPCL(scan_time_, imu, distortedPointCloud, cloudInfoMsg,
                        outlierPointCloud);

  // Clear all measurements before the current time stamp
  imuBuf_.clean(estimator->getTime());
  pclBuf_.clean(estimator->getTime());
  cloudInfoBuf_.clean(estimator->getTime());
  outlierBuf_.clean(estimator->getTime());

  return true;
}

void LinsFusion::performStateEstimation() {
  if (imuBuf_.empty() || pclBuf_.empty() || cloudInfoBuf_.empty() ||
      outlierBuf_.empty())
    return;

  // 经过前面的if选择, 此时既有imu数据,也有点云数据

  // 是否已经初始化,只在处理第一帧点云时进入
  if (!estimator->isInitialized()) {
    processFirstPointCloud();
    return;
  }

  // Iterate all PCL measurements in the buffer
  // 获取最新一帧点云的时间戳
  pclBuf_.getLastTime(last_scan_time_);
  // 点云队列不为空, 且 最新的点云是新的,即在上次状态估计以后获取的
  while (!pclBuf_.empty() && estimator->getTime() < last_scan_time_) {
    TicToc ts_total;
    if (!processPointClouds()) break;
    double time_total = ts_total.toc();
    duration_ = (duration_ * scan_counter_ + time_total) / (scan_counter_ + 1);
    scan_counter_++;
    // ROS_INFO_STREAM("Pure-odometry processing time: " << duration_);
    publishTopics();

    // if (VERBOSE) {
    //   cout << "ba: " << estimator->globalState_.ba_.transpose() << endl;
    //   cout << "bw: " << estimator->globalState_.bw_.transpose() << endl;
    //   cout << "gw: " << estimator->globalState_.gn_.transpose() << endl;
    //   duration_ = (duration_ * scan_counter_ + time_total) / (scan_counter_ +
    //   1); scan_counter_++; ROS_INFO_STREAM(); cout << "Odometry: time: " <<
    //   duration_ << endl;
    // }
  }
}

void LinsFusion::alignIMUtoVehicle(const V3D& rpy, const V3D& acc_in,
                                   const V3D& gyr_in, V3D& acc_out,
                                   V3D& gyr_out) {
  M3D R = rpy2R(rpy);
  acc_out = R.transpose() * acc_in;
  gyr_out = R.transpose() * gyr_in;
}

void LinsFusion::publishOdometryYZX(double timeStamp) {
  laserOdometry.header.frame_id = "camera_init";
  laserOdometry.child_frame_id = "laser_odom";
  laserOdometry.header.stamp = ros::Time().fromSec(timeStamp);
  laserOdometry.pose.pose.orientation.x = estimator->globalStateYZX_.qbn_.x();
  laserOdometry.pose.pose.orientation.y = estimator->globalStateYZX_.qbn_.y();
  laserOdometry.pose.pose.orientation.z = estimator->globalStateYZX_.qbn_.z();
  laserOdometry.pose.pose.orientation.w = estimator->globalStateYZX_.qbn_.w();
  laserOdometry.pose.pose.position.x = estimator->globalStateYZX_.rn_[0];
  laserOdometry.pose.pose.position.y = estimator->globalStateYZX_.rn_[1];
  laserOdometry.pose.pose.position.z = estimator->globalStateYZX_.rn_[2];
  pubLaserOdometry.publish(laserOdometry);

  tf::TransformBroadcaster tfBroadcaster;
  tf::StampedTransform laserOdometryTrans;
  laserOdometryTrans.frame_id_ = "camera_init";
  laserOdometryTrans.child_frame_id_ = "laser_odom";
  laserOdometryTrans.stamp_ = ros::Time().fromSec(timeStamp);
  laserOdometryTrans.setRotation(tf::Quaternion(
      estimator->globalStateYZX_.qbn_.x(), estimator->globalStateYZX_.qbn_.y(),
      estimator->globalStateYZX_.qbn_.z(),
      estimator->globalStateYZX_.qbn_.w()));
  laserOdometryTrans.setOrigin(tf::Vector3(estimator->globalStateYZX_.rn_[0],
                                           estimator->globalStateYZX_.rn_[1],
                                           estimator->globalStateYZX_.rn_[2]));
  tfBroadcaster.sendTransform(laserOdometryTrans);

  fout_evo<<timeStamp<<" "
              <<laserOdometry.pose.pose.position.z<<" "
              <<laserOdometry.pose.pose.position.x<<" "
              <<laserOdometry.pose.pose.position.y<<" "
              <<laserOdometry.pose.pose.orientation.z<<" "
              <<laserOdometry.pose.pose.orientation.x<<" "
              <<laserOdometry.pose.pose.orientation.y<<" "
              <<laserOdometry.pose.pose.orientation.w<<endl;

  // cout<<"time of odometry is "<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC <<"ms"<<endl<<endl;
  // time_stt = clock();
}

// void LinsFusion::performImuBiasEstimation() {
//   Imu imu;
//   while (imuBuf_.getSize() != 0) {
//     imuBuf_.getFirstMeas(imu);
//     ba_tmp_ += imu.acc - V3D(0, 0, G0);
//     bw_tmp_ += imu.gyr;
//     sample_counter_++;
//     if (sample_counter_ == AVERAGE_NUMS) {
//       ba_init_ = ba_tmp_ * (1. / sample_counter_);
//       bw_init_ = bw_tmp_ * (1. / sample_counter_);
//       isImuCalibrated = true;
//       ROS_INFO_STREAM("Estimated IMU acceleration bias: \n "
//                       << ba_init_.transpose() << " and gyroscope bias: \n"
//                       << bw_init_.transpose());
//       ba_tmp_.setZero();
//       bw_tmp_.setZero();
//       sample_counter_ = 0;
//       break;
//     }
//     imuBuf_.clean(imu.time);
//     pclBuf_.clean(imu.time);
//     outlierBuf_.clean(imu.time);
//     cloudInfoBuf_.clean(imu.time);
//   }
// }

}  // namespace fusion
