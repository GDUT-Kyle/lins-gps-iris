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

#ifndef INCLUDE_STATEESTIMATOR_HPP_
#define INCLUDE_STATEESTIMATOR_HPP_

#include <integrationBase.h>
#include <math_utils.h>
#include <parameters.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/Imu.h>
#include <tf/transform_broadcaster.h>
#include <tic_toc.h>

#include <KalmanFilter.hpp>
#include <algorithm>
#include <boost/shared_ptr.hpp>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <sensor_utils.hpp>
#include <vector>

#include "cloud_msgs/cloud_info.h"
#include "integrationBase.h"

using namespace Eigen;
using namespace std;
using namespace math_utils;
using namespace sensor_utils;
using namespace parameter;
using namespace filter;

namespace fusion {

const int LINE_NUM_ = 16;
const int SCAN_NUM_ = 1800;

struct Smooth {
  Smooth() {
    value = 0.0;
    ind = 0;
  }
  double value;
  size_t ind;
};

struct byValue {
  bool operator()(Smooth const& left, Smooth const& right) {
    return left.value < right.value;
  }
};

// Scan Class stores all kinds of information of a point cloud, including
// the whole point cloud, its smoothness, timestamp, and features.
class Scan {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Scan() : id_(scan_counter_++) {
    distPointCloud_.reset(new pcl::PointCloud<PointType>());
    undistPointCloud_.reset(new pcl::PointCloud<PointType>());
    outlierPointCloud_.reset(new pcl::PointCloud<PointType>());
    cornerPointsSharp_.reset(new pcl::PointCloud<PointType>());
    cornerPointsLessSharp_.reset(new pcl::PointCloud<PointType>());
    surfPointsFlat_.reset(new pcl::PointCloud<PointType>());
    surfPointsLessFlat_.reset(new pcl::PointCloud<PointType>());
    cloudInfo_.reset(new cloud_msgs::cloud_info());

    cornerPointsLessSharpYZX_.reset(new pcl::PointCloud<PointType>());
    surfPointsLessFlatYZX_.reset(new pcl::PointCloud<PointType>());
    outlierPointCloudYZX_.reset(new pcl::PointCloud<PointType>());

    cloudCurvature_.resize(LINE_NUM * SCAN_NUM);
    cloudSmoothness_.resize(LINE_NUM * SCAN_NUM);

    reset();
  }

  ~Scan() {
    distPointCloud_.reset();
    undistPointCloud_.reset();
    outlierPointCloud_.reset();
    cornerPointsSharp_.reset();
    cornerPointsLessSharp_.reset();
    surfPointsFlat_.reset();
    surfPointsLessFlat_.reset();
    cloudInfo_.reset();

    cornerPointsLessSharpYZX_.reset();
    surfPointsLessFlatYZX_.reset();
    outlierPointCloudYZX_.reset();
  }

  void reset() {
    time_ = 0.0;

    distPointCloud_->clear();
    undistPointCloud_->clear();
    outlierPointCloud_->clear();
    cornerPointsSharp_->clear();
    cornerPointsLessSharp_->clear();
    surfPointsFlat_->clear();
    surfPointsLessFlat_->clear();

    cornerPointsLessSharpYZX_->clear();
    surfPointsLessFlatYZX_->clear();
    outlierPointCloudYZX_->clear();

    cloudCurvature_.assign(LINE_NUM * SCAN_NUM, 0.0);
    cloudSmoothness_.assign(LINE_NUM * SCAN_NUM, Smooth());
  }

  void setPointCloud(double time,
                     pcl::PointCloud<PointType>::Ptr distPointCloud,
                     cloud_msgs::cloud_info cloudInfo,
                     pcl::PointCloud<PointType>::Ptr outlierPointCloud) {
    distPointCloud_ = distPointCloud;
    *(cloudInfo_) = cloudInfo;
    outlierPointCloud_ = outlierPointCloud;
    time_ = time;
  }

 public:
  // !@ScanInfo
  static int scan_counter_;
  int id_;
  double time_;

  // !@PointCloud
  pcl::PointCloud<PointType>::Ptr distPointCloud_;
  pcl::PointCloud<PointType>::Ptr undistPointCloud_;
  pcl::PointCloud<PointType>::Ptr outlierPointCloud_;
  cloud_msgs::cloud_info::Ptr cloudInfo_;

  // !@PclFeatures
  std::vector<double> cloudCurvature_;
  std::vector<Smooth> cloudSmoothness_;

  int cloudNeighborPicked_[LINE_NUM_ * SCAN_NUM_];
  int cloudLabel_[LINE_NUM_ * SCAN_NUM_];

  pcl::PointCloud<PointType>::Ptr cornerPointsSharp_;
  pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp_;
  pcl::PointCloud<PointType>::Ptr surfPointsFlat_;
  pcl::PointCloud<PointType>::Ptr surfPointsLessFlat_;

  pcl::PointCloud<PointType>::Ptr cornerPointsLessSharpYZX_;
  pcl::PointCloud<PointType>::Ptr surfPointsLessFlatYZX_;
  pcl::PointCloud<PointType>::Ptr outlierPointCloudYZX_;
};
typedef shared_ptr<Scan> ScanPtr;  // Define a pointer class for Scan class

// StateEstimator Class implement a iterative-ESKF, including state propagation
// and update.
class StateEstimator {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  enum FusionStatus {
    STATUS_INIT = 0,
    STATUS_FIRST_SCAN = 1,
    STATUS_SECOND_SCAN = 2,
    STATUS_RUNNING = 3,
    STATUS_RESET = 4,
  };

  StateEstimator() {
    filter_ = new StatePredictor();

    // Initialize KD tree and downsize filter
    downSizeFilter_.setLeafSize(0.2, 0.2, 0.2);
    kdtreeCorner_.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeSurf_.reset(new pcl::KdTreeFLANN<PointType>());
    scan_new_.reset(new Scan());
    scan_last_.reset(new Scan());

    keypoints_.reset(new pcl::PointCloud<PointType>());
    jacobians_.reset(new pcl::PointCloud<PointType>());
    keypointCorns_.reset(new pcl::PointCloud<PointType>());
    keypointSurfs_.reset(new pcl::PointCloud<PointType>());
    jacobianCoffCorns.reset(new pcl::PointCloud<PointType>());
    jacobianCoffSurfs.reset(new pcl::PointCloud<PointType>());

    surfPointsLessFlatScan.reset(new pcl::PointCloud<PointType>());
    surfPointsLessFlatScanDS.reset(new pcl::PointCloud<PointType>());

    pointSelCornerInd.resize(LINE_NUM * SCAN_NUM);
    pointSearchCornerInd1.resize(LINE_NUM * SCAN_NUM);
    pointSearchCornerInd2.resize(LINE_NUM * SCAN_NUM);
    pointSelSurfInd.resize(LINE_NUM * SCAN_NUM);
    pointSearchSurfInd1.resize(LINE_NUM * SCAN_NUM);
    pointSearchSurfInd2.resize(LINE_NUM * SCAN_NUM);
    pointSearchSurfInd3.resize(LINE_NUM * SCAN_NUM);

    globalState_.setIdentity();
    globalStateYZX_.setIdentity();

    // Rotation matrics between XYZ convention and YZX-convention
    R_yzx_to_xyz << 0., 0., 1., 1., 0., 0., 0., 1., 0.;
    // 旋转矩阵转置就是取逆
    R_xyz_to_yzx = R_yzx_to_xyz.transpose();
    Q_yzx_to_xyz = R_yzx_to_xyz;
    Q_xyz_to_yzx = R_xyz_to_yzx;

    // gravity_feedback << 0, 0, -G0;

    status_ = STATUS_INIT;
  }

  ~StateEstimator() {
    delete filter_;
    delete preintegration_;
  }

  inline const double getTime() const { return filter_->time_; }
  inline bool isInitialized() const { return status_ != STATUS_INIT; }

  /********Relative Variables*********/
  V3D pos_;
  V3D vel_;
  M3D quad_;
  V3D acc_0_;
  V3D gyr_0_;
  /***********************************/
  void processImu(double dt, const V3D& acc, const V3D& gyr) {
    switch (status_) {
      case STATUS_INIT:
        break;
      case STATUS_FIRST_SCAN:
        preintegration_->push_back(dt, acc, gyr);
        filter_->time_ += dt;
        acc_0_ = acc;
        gyr_0_ = gyr;
        break;
      case STATUS_RUNNING:
        // 预积分更新状态[pos,vel,att,acc,gyr,gra]
        filter_->predict(dt, acc, gyr, true);
        break;
      default:
        break;
    }

    // For no use here. Just propagte IMU measurements for testing
    V3D un_acc_0_ = quad_ * (acc_0_ - INIT_BA) + filter_->state_.gn_;
    V3D un_gyr = 0.5 * (gyr_0_ + gyr) - INIT_BW;
    quad_ *= math_utils::deltaQ(un_gyr * dt).toRotationMatrix();
    V3D un_acc_1 = quad_ * (acc - INIT_BA) + filter_->state_.gn_;
    V3D un_acc = 0.5 * (un_acc_0_ + un_acc_1);
    pos_ += dt * vel_ + 0.5 * dt * dt * un_acc;
    vel_ += dt * un_acc;

    acc_0_ = acc;
    gyr_0_ = gyr;
  }

  /********Relative Variables*********/
  double duration_fea_ = 0;
  double duration_opt_ = 0;
  double num_of_edge_ = 0;
  double num_of_surf_ = 0;
  int lidar_counter_ = 0;
  /***********************************/
  void processPCL(double time, const Imu& imu,
                  pcl::PointCloud<PointType>::Ptr distortedPointCloud,
                  cloud_msgs::cloud_info cloudInfo,
                  pcl::PointCloud<PointType>::Ptr outlierPointCloud) {
    TicToc ts_fea;  // Calculate the time used in feature extraction
    // 将这些数据存入scan_new_对象中
    scan_new_->setPointCloud(time, distortedPointCloud, cloudInfo,
                             outlierPointCloud);
    // 运动畸变矫正，scan_new_->undistPointCloud_的intensity存储了每个点的时间比例
    undistortPcl(scan_new_);
    // 像LOAM一样计算曲率
    calculateSmoothness(scan_new_);
    // 标记被遮挡点
    markOccludedPoints(scan_new_);
    // 提取特征点
    extractFeatures(scan_new_);
    imu_last_ = imu;
    double time_fea = ts_fea.toc();

    TicToc ts_opt;  // Calculate the time used in state estimation
    // 判断系统状态
    switch (status_) {
      case STATUS_INIT:
        // 初始化滤波器
        if (processFirstScan()) status_ = STATUS_FIRST_SCAN;
        break;
      case STATUS_FIRST_SCAN:
        // 初始化状态和误差状态
        if (processSecondScan())
          status_ = STATUS_RUNNING;
        else
          status_ = STATUS_INIT;
        break;
      case STATUS_RUNNING:
        if (!processScan()) status_ = STATUS_RUNNING;
        break;
    }
    double time_opt = ts_opt.toc();

    // if (VERBOSE) {
    //   duration_fea_ =
    //       (duration_fea_ * lidar_counter_ + time_fea) / (lidar_counter_ + 1);
    //   duration_opt_ =
    //       (duration_opt_ * lidar_counter_ + time_opt) / (lidar_counter_ + 1);
    //   num_of_edge_ = (num_of_edge_ * lidar_counter_ +
    //                   scan_last_->cornerPointsLessSharpYZX_->points.size()) /
    //                  (lidar_counter_ + 1);
    //   num_of_surf_ = (num_of_surf_ * lidar_counter_ +
    //                   scan_last_->surfPointsLessFlatYZX_->points.size()) /
    //                  (lidar_counter_ + 1);
    //   lidar_counter_++;

    //   // cout << "Feature Extraction: time: " << duration_fea_ << endl;
    //   // cout << "Feature Extraction: corners: " << num_of_edge_
    //   //      << ", surfs: " << num_of_surf_ << endl;
    //   // cout << "State Estimation: time: " << duration_opt_ << endl;
    // }
  }

  // Initialize the Kalman filter and KD tree
  bool processFirstScan() {
    // 判断特征点数目是否足够去初始化
    if (scan_new_->cornerPointsLessSharp_->points.size() < 10 ||
        scan_new_->surfPointsLessFlat_->points.size() < 100) {
      ROS_WARN("Wait for more features for initialization...");
      scan_new_.reset(new Scan());
      return false;
    }

    // Initialize the Kalman filter
    // 注意论文中的公式有误,应该查找论文《Estimation techniques for low-cost inertial navigation》公式(3.39)
    // IMU误差模型的线性连续时间系统方程,根据他可以推导出IMU误差模型的预测更新方程
    // F-18*18, 论文公式(6), IMU测量的误差状态传递矩阵, 此处状态量比论文描述的多了一个三维的重力加速度g
    Fk_.resize(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_);
    // G-18*12, 论文公式(7), IMU测量的误差状态关于时间的噪声雅克比
    Gk_.resize(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_NOISE_);
    // P-18*18, 论文公式(9), IMU测量的误差状态预测的协方差矩阵
    Pk_.resize(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_);
    // Q-12*12, 系统噪声的协方差矩阵
    Qk_.resize(GlobalState::DIM_OF_NOISE_, GlobalState::DIM_OF_NOISE_);
    // 用于计算后验协方差的中间变量
    IKH_.resize(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_);

    Fk_.setIdentity();
    Gk_.setZero();
    Pk_.setZero();
    Qk_.setZero();

    // Set the relative transform to identity
    linState_.setIdentity();

    // Initialize IMU preintegration variable
    // 初始化imu的预积分变量，记录首次测量的加速度和角速度，INIT_BA和INIT_BW分别是加速度计和陀螺仪的测量噪声
    preintegration_ = new integration::IntegrationBase(
        imu_last_.acc, imu_last_.gyr, INIT_BA, INIT_BW);

    // Initialize position, linear velocity, angular velocity, acceleration bias, gyroscope bias by zeros
    // 初始化滤波器
    filter_->initialization(scan_new_->time_, V3D(0, 0, 0), V3D(0, 0, 0),
                            V3D(0, 0, 0), V3D(0, 0, 0), imu_last_.acc,
                            imu_last_.gyr);

    // 用特征点构建kd树,用于下一帧点云寻找correspondence
    kdtreeCorner_->setInputCloud(scan_new_->cornerPointsLessSharp_);
    kdtreeSurf_->setInputCloud(scan_new_->surfPointsLessFlat_);

    // 位置
    pos_.setZero();
    // 速度
    vel_.setZero();
    // 旋转矩阵
    quad_.setIdentity();

    // Slide the point cloud from scan_new_ to scan_last_
    scan_last_.swap(scan_new_);
    scan_new_.reset(new Scan());

    return true;
  }

  // Calculate initial velocity and IMU biases using two consecutive frames and
  // IMU preintegration results
  // 应用两个连续帧和Imu预积分结果计算初始速度和imu误差
  bool processSecondScan() {
    // 特征点云数目是否足够
    if (scan_new_->cornerPointsLessSharp_->points.size() < 10 ||
        scan_new_->surfPointsLessFlat_->points.size() < 100) {
      ROS_WARN("Wait for more features for initialization...");
      scan_new_.reset(new Scan());
      return false;
    }

    // Calculate relative transform, linState_, using ICP method
    V3D pl;
    Q4D ql;
    V3D v0, v1, ba0, bw0;
    ba0.setZero();
    // 预积分后得到的attitude误差状态
    ql = preintegration_->delta_q;
    // 预积分后得到position误差状态,这里要去除重力加速度的影响和加速度计误差状态的影响
    // ba0已经setZero了,还有必要加进去吗
    pl = preintegration_->delta_p +
        //  在midPointIntegration函数预积分计算preintegration_->delta_p没有去掉重力加速度影响,因此这里要补上
         0.5 * linState_.gn_ * preintegration_->sum_dt *
             preintegration_->sum_dt -
         0.5 * ba0 * preintegration_->sum_dt * preintegration_->sum_dt;
    // 其实ql和pl就是Imu预积分预测的帧间变换，再将其作为icp迭代初值可以加速求解
    // 通过ICP计算帧间位姿变换，存储于pl,ql
    estimateTransform(scan_last_, scan_new_, pl, ql);

    // Calculate initial state using relative transform calculated by point
    // clouds and that by IMU preintegration
    // 设置初始的状态
    estimateInitialState(pl, ql, v0, v1, ba0, bw0);

    // Initialize the Kalman filter by estimated values
    // 初始化卡尔曼滤波器的状态向量
    V3D r1 = pl;
    filter_->initialization(scan_new_->time_, r1, v1, ba0, bw0, imu_last_.acc,
                            imu_last_.gyr);

    double roll_init, pitch_init, yaw_init = deg2rad(0.0);
    // Calculate rough roll and pitch angles using IMU measurements
    // 利用imu测量粗略计算车体的roll和pitch
    calculateRPfromGravity(imu_last_.acc - ba0, roll_init, pitch_init);

    // Initialize the global state, e.g., position, velocity, and orientation
    // represented in the original frame (the first-scan-frame)
    // 初始化全局坐标系下的状态向量
    globalState_ = GlobalState(
        r1, v1, rpy2Quat(V3D(roll_init, pitch_init, yaw_init)), ba0, bw0);

    // Use relative transorm linState_ to undistort point cloud (under the
    // constant-speed assumption)
    // 利用前面的帧间位姿变换矫正点云的运动畸变，这里的前提是假设它是匀速运动模型
    updatePointCloud();

    // 更新帧
    scan_last_.swap(scan_new_);
    scan_new_.reset(new Scan());

    return true;
  }

  void correctRollPitch(const double& roll, const double& pitch) {
    V3D rpy = math_utils::Q2rpy(globalState_.qbn_);
    Q4D quad = math_utils::rpy2Quat(V3D(roll, pitch, rpy[2]));
    globalState_.qbn_ = quad;
  }

  void correctOrientation(const Q4D& quad) { globalState_.qbn_ = quad; }

  bool processScan() {
    // 特征点数目要足够多才可信
    if (scan_new_->cornerPointsLessSharp_->points.size() <= 5 ||
        scan_new_->surfPointsLessFlat_->points.size() <= 10) {
      ROS_WARN("Insufficient features...State estimation fails.");
      return false;
    }

    // Update states
    // 通过IEKF更新误差状态
    performIESKF();
    // Update global transform by estimated relative transform
    // 将误差状态积累到状态量上
    integrateTransformation();
    // 清空滤波器中那些点云占用的空间
    filter_->reset(1);

    double roll, pitch;
    // Because the estimated gravity is represented in the b-frame, we can
    // directly solve more accurate roll and pitch angles to correct the global
    // state
    // 通过重力加速度矫正roll和pitch
    calculateRPfromGravity(filter_->state_.gn_, roll, pitch);
    correctRollPitch(roll, pitch);

    // Undistort point cloud using estimated relative transform
    updatePointCloud();

    // Slide the new scan to last scan
    scan_last_.swap(scan_new_);
    scan_new_.reset(new Scan());

    return true;
  }

  void performIESKF() {
    // Store current state and perform initialization
    // 预测协方差
    Pk_ = filter_->covariance_;
    // 预测的状态
    GlobalState filterState = filter_->state_;
    linState_ = filterState;

    double residualNorm = 1e6;
    bool hasConverged = false;
    bool hasDiverged = false;
    const unsigned int DIM_OF_STATE = GlobalState::DIM_OF_STATE_;
    // 迭代NUM_ITER次
    for (int iter = 0; iter < NUM_ITER && !hasConverged && !hasDiverged;
         iter++) {
      keypointSurfs_->clear();
      jacobianCoffSurfs->clear();
      keypointCorns_->clear();
      jacobianCoffCorns->clear();

      // Find corresponding features
      // 计算ICP的jacobian
      findCorrespondingSurfFeatures(scan_last_, scan_new_, keypointSurfs_,
                                    jacobianCoffSurfs, iter);
      if (keypointSurfs_->points.size() < 10) {
        if (VERBOSE) {
          ROS_WARN("Insufficient matched surfs...");
        }
      }
      findCorrespondingCornerFeatures(scan_last_, scan_new_, keypointCorns_,
                                      jacobianCoffCorns, iter);
      if (keypointCorns_->points.size() < 5) {
        if (VERBOSE) {
          ROS_WARN("Insufficient matched corners...");
        }
      }

      // Sum up jocobians and residuals
      keypoints_->clear();
      jacobians_->clear();
      // 组建新的关键点向量
      (*keypoints_) += (*keypointSurfs_);
      (*keypoints_) += (*keypointCorns_);
      // 组建新的雅克比
      (*jacobians_) += (*jacobianCoffSurfs);
      (*jacobians_) += (*jacobianCoffCorns);

      // Memery allocation
      // 配置内存空间
      const unsigned int DIM_OF_MEAS = keypoints_->points.size();
      residual_.resize(DIM_OF_MEAS);
      // ICP关于误差状态delta x的雅克比矩阵
      Hk_.resize(DIM_OF_MEAS, DIM_OF_STATE);
      // lidar测量噪声的协方差矩阵
      Rk_.resize(DIM_OF_MEAS, DIM_OF_MEAS);
      // IEKF的卡尔曼增益
      Kk_.resize(DIM_OF_STATE, DIM_OF_MEAS);
      // 用于计算Kk的中间变量
      Py_.resize(DIM_OF_MEAS, DIM_OF_MEAS);
      Pyinv_.resize(DIM_OF_MEAS, DIM_OF_MEAS);

      Hk_.setZero();
      // 四元数转欧拉角
      V3D axis = Quat2axis(linState_.qbn_);
      // 遍历每个特征点
      for (int i = 0; i < DIM_OF_MEAS; ++i) {
        // Point represented in 2-frame (e.g., the end frame) in a
        // xyz-convention
        // 特征点
        V3D P2xyz(keypoints_->points[i].x, keypoints_->points[i].y,
                  keypoints_->points[i].z);
        // 该特征点对应的雅克比
        V3D coff_xyz(jacobians_->points[i].x, jacobians_->points[i].y,
                     jacobians_->points[i].z);
        // 残差，就是点面距离或点线距离
        // residual_ = h(x_i)-z
        residual_(i) = LIDAR_SCALE * jacobians_->points[i].intensity;

        // TODO:这里怎么求？不懂
        Hk_.block<1, 3>(i, GlobalState::att_) =
            coff_xyz.transpose() *
            (-linState_.qbn_.toRotationMatrix() * skew(P2xyz)) *
            Rinvleft(-axis); // Rinvleft(a)是求向量a的左伪逆
        Hk_.block<1, 3>(i, GlobalState::pos_) =
            coff_xyz.transpose() * M3D::Identity();
      }

      // Set the measurement covariance matrix
      // 设置测量噪声协方差矩阵，就是论文中的M_k
      VXD cov = VXD::Zero(DIM_OF_MEAS);
      for (int i = 0; i < DIM_OF_MEAS; ++i) {
        cov[i] = LIDAR_STD * LIDAR_STD;
      }
      Rk_ = cov.asDiagonal();

      // Kalman filter update. Details can be referred to ROVIO
      // 论文公式(13),但是论文公式里的J_k(ICP关于测量噪声的雅克比)在这里省略了，大概是因为其影响不大
      Py_ =
          Hk_ * Pk_ * Hk_.transpose() + Rk_;  // S = H * P * H.transpose() + R;
      Pyinv_.setIdentity();                   // solve Ax=B
      Py_.llt().solveInPlace(Pyinv_);
      Kk_ = Pk_ * Hk_.transpose() * Pyinv_;  // K = P*H.transpose()*S.inverse()

      // 论文公式(14),不太懂
      // difVecLinInv_ = filterState (-) linState_ ~~ difVecLinInv_ = \hat{x}_last - x_i
      // 注意: difVecLinInv_ = - \Delta x
      filterState.boxMinus(linState_, difVecLinInv_);
      // 这里好像跟公式不对应????
      // 解答: residual_ = h(x_i)-z, 带入方程中就与公式一致了
      updateVec_ = -Kk_ * (residual_ + Hk_ * difVecLinInv_) + difVecLinInv_;

      // Divergence determination
      // 迭代发散判断
      bool hasNaN = false;
      for (int i = 0; i < updateVec_.size(); i++) {
        if (isnan(updateVec_[i])) {
          updateVec_[i] = 0;
          hasNaN = true;
        }
      }
      if (hasNaN == true) {
        ROS_WARN("System diverges Because of NaN...");
        hasDiverged = true;
        break;
      }

      // Check whether the filter converges
      // 检查滤波器是否迭代收敛
      if (residual_.norm() > residualNorm * 10) {
        ROS_WARN("System diverges...");
        hasDiverged = true;
        break;
      }

      // Update the state
      // 论文公式(15)
      // linState_ = linState_ (+) updateVec_  ~~  x_{i+1} = x_{i} + updateVec_
      linState_.boxPlus(updateVec_, linState_);

      updateVecNorm_ = updateVec_.norm();
      if (updateVecNorm_ <= 1e-2) {
        hasConverged = true;
      }

      residualNorm = residual_.norm();
    }

    // If diverges, swtich to traditional ICP method to get a rough relative
    // transformation. Otherwise, update the error-state covariance matrix
    // 如果迭代发散，则采用icp估算位姿，那就是跟lego-loam一样了
    if (hasDiverged == true) {
      ROS_WARN("======Using ICP Method======");
      V3D t = filterState.rn_;
      Q4D q = filterState.qbn_;
      estimateTransform(scan_last_, scan_new_, t, q);
      filterState.rn_ = t;
      filterState.qbn_ = q;
      filter_->update(filterState, Pk_);
    } else {
      // Update only one time
      // 论文公式(16)，更新后验协方差
      IKH_ = Eigen::Matrix<double, 18, 18>::Identity() - Kk_ * Hk_;
      Pk_ = IKH_ * Pk_ * IKH_.transpose() + Kk_ * Rk_ * Kk_.transpose();
      // 0.5 * P_k * P_k^T形成对称矩阵，近似P_k,但是它是半正定的，更稳定
      enforceSymmetry(Pk_);
      // 更新滤波器状态和协方差
      filter_->update(linState_, Pk_);
    }
  }

  void calculateRPfromGravity(const V3D& fbib, double& roll, double& pitch) {
    pitch = -sign(fbib.z()) * asin(fbib.x() / G0);
    roll = sign(fbib.z()) * asin(fbib.y() / G0);
  }

  // Update the gloabl state by the new relative transformation
  // 将误差状态积累到状态量上
  void integrateTransformation() {
    GlobalState filterState = filter_->state_;
    globalState_.rn_ = globalState_.qbn_ * filterState.rn_ + globalState_.rn_;
    globalState_.qbn_ = globalState_.qbn_ * filterState.qbn_;
    globalState_.vn_ =
        globalState_.qbn_ * filterState.qbn_.inverse() * filterState.vn_;
    globalState_.ba_ = filterState.ba_;
    globalState_.bw_ = filterState.bw_;
    globalState_.gn_ = globalState_.qbn_ * filterState.gn_;
  }

  void undistortPcl(ScanPtr scan) {
    bool halfPassed = false;
    scan->undistPointCloud_->clear();
    pcl::PointCloud<PointType>::Ptr distPointCloud = scan->distPointCloud_;
    cloud_msgs::cloud_info::Ptr segInfo = scan->cloudInfo_;
    int size = distPointCloud->points.size();
    PointType point;
    // 遍历该帧点云每个点
    for (int i = 0; i < size; i++) {
      // If LiDAR frame does not align with Vehic frame, we transform the point
      // cloud to the vehicle frame
      // 通过外参,将lidar点云转到车体坐标系下
      rotatePoint(&distPointCloud->points[i], &point);

      double ori = -atan2(point.y, point.x);
      if (!halfPassed) {
        if (ori < segInfo->startOrientation - M_PI / 2)
          ori += 2 * M_PI;
        else if (ori > segInfo->startOrientation + M_PI * 3 / 2)
          ori -= 2 * M_PI;

        if (ori - segInfo->startOrientation > M_PI) halfPassed = true;
      } else {
        ori += 2 * M_PI;

        if (ori < segInfo->endOrientation - M_PI * 3 / 2)
          ori += 2 * M_PI;
        else if (ori > segInfo->endOrientation + M_PI / 2)
          ori -= 2 * M_PI;
      }
      // 计算该点在帧中所处的时间位置，用于后续线性插值
      double relTime =
          (ori - segInfo->startOrientation) / segInfo->orientationDiff;
      point.intensity =
          int(distPointCloud->points[i].intensity) + SCAN_PERIOD * relTime;

      scan->undistPointCloud_->push_back(point);
    }
  }

  // 像LOAM一样计算曲率
  void calculateSmoothness(ScanPtr scan) {
    int cloudSize = scan->undistPointCloud_->points.size();
    cloud_msgs::cloud_info::Ptr segInfo = scan->cloudInfo_;
    for (int i = 5; i < cloudSize - 5; i++) {
      double diffRange = segInfo->segmentedCloudRange[i - 5] +
                         segInfo->segmentedCloudRange[i - 4] +
                         segInfo->segmentedCloudRange[i - 3] +
                         segInfo->segmentedCloudRange[i - 2] +
                         segInfo->segmentedCloudRange[i - 1] -
                         segInfo->segmentedCloudRange[i] * 10 +
                         segInfo->segmentedCloudRange[i + 1] +
                         segInfo->segmentedCloudRange[i + 2] +
                         segInfo->segmentedCloudRange[i + 3] +
                         segInfo->segmentedCloudRange[i + 4] +
                         segInfo->segmentedCloudRange[i + 5];
      // 存储曲率
      scan->cloudCurvature_[i] = diffRange * diffRange;

      // 标记近邻被选取点
      scan->cloudNeighborPicked_[i] = 0;
      scan->cloudLabel_[i] = 0;
      scan->cloudSmoothness_[i].value = scan->cloudCurvature_[i];
      scan->cloudSmoothness_[i].ind = i;
    }
  }

  // 标记被遮挡点，参考LOAM论文
  void markOccludedPoints(ScanPtr scan) {
    int cloudSize = scan->undistPointCloud_->points.size();
    cloud_msgs::cloud_info::Ptr segInfo = scan->cloudInfo_;
    // 涉及到与后一个点i+1差值，所以cloudSize-6不是-5
    for (int i = 5; i < cloudSize - 6; ++i) {
      // 点的深度
      float depth1 = segInfo->segmentedCloudRange[i];
      float depth2 = segInfo->segmentedCloudRange[i + 1];
      int columnDiff = std::abs(int(segInfo->segmentedCloudColInd[i + 1] -
                                    segInfo->segmentedCloudColInd[i]));
      // 保证两点在同一ring上
      if (columnDiff < 10) {
        // 两点之间距离大于0.3，则认为是不可信点
        if (depth1 - depth2 > 0.3) {
          scan->cloudNeighborPicked_[i - 5] = 1;
          scan->cloudNeighborPicked_[i - 4] = 1;
          scan->cloudNeighborPicked_[i - 3] = 1;
          scan->cloudNeighborPicked_[i - 2] = 1;
          scan->cloudNeighborPicked_[i - 1] = 1;
          scan->cloudNeighborPicked_[i] = 1;
        } else if (depth2 - depth1 > 0.3) {
          scan->cloudNeighborPicked_[i + 1] = 1;
          scan->cloudNeighborPicked_[i + 2] = 1;
          scan->cloudNeighborPicked_[i + 3] = 1;
          scan->cloudNeighborPicked_[i + 4] = 1;
          scan->cloudNeighborPicked_[i + 5] = 1;
          scan->cloudNeighborPicked_[i + 6] = 1;
        }
      }
      // point[i-1]和point[i+1]分别到point[i]的距离不能大于0.02×point[i].range,否则标记point[i]为不可选
      float diff1 = std::abs(segInfo->segmentedCloudRange[i - 1] -
                             segInfo->segmentedCloudRange[i]);
      float diff2 = std::abs(segInfo->segmentedCloudRange[i + 1] -
                             segInfo->segmentedCloudRange[i]);
      if (diff1 > 0.02 * segInfo->segmentedCloudRange[i] &&
          diff2 > 0.02 * segInfo->segmentedCloudRange[i])
        scan->cloudNeighborPicked_[i] = 1;
    }
  }

  /********Relative Variables*********/
  pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan;
  pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScanDS;
  /***********************************/
  void extractFeatures(ScanPtr scan) {
    cloud_msgs::cloud_info::Ptr segInfo = scan->cloudInfo_;

    scan->cornerPointsSharp_->clear(); // 边缘点
    scan->cornerPointsLessSharp_->clear(); // 大于阈值但不是边缘点
    scan->surfPointsFlat_->clear(); // 平面点
    scan->surfPointsLessFlat_->clear(); // 小于阈值但不是平面点

    // 遍历每条线
    for (int i = 0; i < LINE_NUM; i++) {
      surfPointsLessFlatScan->clear();

      // 将每条线平均分成6块
      for (int j = 0; j < 6; j++) {
        // 每块的start point index
        int sp = (segInfo->startRingIndex[i] * (6 - j) +
                  segInfo->endRingIndex[i] * j) / 6;
        // 每块的end point index
        int ep = (segInfo->startRingIndex[i] * (5 - j) +
                  segInfo->endRingIndex[i] * (j + 1)) /
                     6 - 1;

        // start和end不属于同一条线
        if (sp >= ep) continue;

        // 块内的点按曲率大小排列
        std::sort(scan->cloudSmoothness_.begin() + sp,
                  scan->cloudSmoothness_.begin() + ep, byValue());

        int largestPickedNum = 0;
        for (int k = ep; k >= sp; k--) {
          int ind = scan->cloudSmoothness_[k].ind;
          // 挑选每个六等分段的曲率大于EDGE_THRESHOLD且没被筛选的且不属于地面的点
          if (scan->cloudNeighborPicked_[ind] == 0 &&
              scan->cloudCurvature_[ind] > EDGE_THRESHOLD &&
              segInfo->segmentedCloudGroundFlag[ind] == false) {
            largestPickedNum++;
            // 每块边缘点不能超过2个
            if (largestPickedNum <= 2) {
              scan->cloudLabel_[ind] = 2;
              // 压入cornerPointsSharp_和cornerPointsLessSharp_
              scan->cornerPointsSharp_->push_back(
                  scan->undistPointCloud_->points[ind]);
              scan->cornerPointsLessSharp_->push_back(
                  scan->undistPointCloud_->points[ind]);
            } else if (largestPickedNum <= 20) {
              scan->cloudLabel_[ind] = 1;
              scan->cornerPointsLessSharp_->push_back(
                  scan->undistPointCloud_->points[ind]);
            } else {
              break;
            }

            scan->cloudNeighborPicked_[ind] = 1;
            for (int l = 1; l <= 5; l++) {
              int columnDiff =
                  std::abs(int(segInfo->segmentedCloudColInd[ind + l] -
                               segInfo->segmentedCloudColInd[ind + l - 1]));
              if (columnDiff > 10) break;
              scan->cloudNeighborPicked_[ind + l] = 1;
            }
            for (int l = -1; l >= -5; l--) {
              int columnDiff =
                  std::abs(int(segInfo->segmentedCloudColInd[ind + l] -
                               segInfo->segmentedCloudColInd[ind + l + 1]));
              if (columnDiff > 10) break;
              scan->cloudNeighborPicked_[ind + l] = 1;
            }
          }
        }

        int smallestPickedNum = 0;
        for (int k = sp; k <= ep; k++) {
          int ind = scan->cloudSmoothness_[k].ind;
          if (scan->cloudNeighborPicked_[ind] == 0 &&
              scan->cloudCurvature_[ind] < SURF_THRESHOLD &&
              segInfo->segmentedCloudGroundFlag[ind] == true) {
            scan->cloudLabel_[ind] = -1;
            scan->surfPointsFlat_->push_back(
                scan->undistPointCloud_->points[ind]);
            smallestPickedNum++;
            if (smallestPickedNum >= 4) {
              break;
            }

            scan->cloudNeighborPicked_[ind] = 1;
            for (int l = 1; l <= 5; l++) {
              int columnDiff =
                  std::abs(int(segInfo->segmentedCloudColInd[ind + l] -
                               segInfo->segmentedCloudColInd[ind + l - 1]));
              if (columnDiff > 10) break;

              scan->cloudNeighborPicked_[ind + l] = 1;
            }
            for (int l = -1; l >= -5; l--) {
              int columnDiff =
                  std::abs(int(segInfo->segmentedCloudColInd[ind + l] -
                               segInfo->segmentedCloudColInd[ind + l + 1]));
              if (columnDiff > 10) break;

              scan->cloudNeighborPicked_[ind + l] = 1;
            }
          }
        }

        for (int k = sp; k <= ep; k++) {
          if (scan->cloudLabel_[k] <= 0) {
            surfPointsLessFlatScan->push_back(
                scan->undistPointCloud_->points[k]);
          }
        }
      }
      surfPointsLessFlatScanDS->clear();
      downSizeFilter_.setInputCloud(surfPointsLessFlatScan);
      downSizeFilter_.filter(*surfPointsLessFlatScanDS);
      *(scan->surfPointsLessFlat_) += *surfPointsLessFlatScanDS;
    }
  }

  void findCorrespondingSurfFeatures(
      ScanPtr lastScan, ScanPtr newScan,
      pcl::PointCloud<PointType>::Ptr keypoints,
      pcl::PointCloud<PointType>::Ptr jacobianCoff, int iterCount) {
    int surfPointsFlatNum = newScan->surfPointsFlat_->points.size();

    // 遍历当前帧中每个planer点
    for (int i = 0; i < surfPointsFlatNum; i++) {
      // point selected
      PointType pointSel;
      PointType coeff, tripod1, tripod2, tripod3;

      // 将点转换到该帧扫描起始位置, 矫正运动畸变
      transformToStart(&newScan->surfPointsFlat_->points[i], &pointSel);

      // 上一帧的less planer点云
      pcl::PointCloud<PointType>::Ptr laserCloudSurfLast =
          lastScan->surfPointsLessFlat_;

      // 每隔ICP_FREQ个点计算一次点面距离, 从而控制运算速度, ICP_FREQ越大, 运算越快, 但是精度也会相对降低
      if (iterCount % ICP_FREQ == 0) {
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        // 从kd树中寻找3个最近邻点
        kdtreeSurf_->nearestKSearch(pointSel, 1, pointSearchInd,
                                    pointSearchSqDis);
        // 存储3个最近邻点
        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;

        // 判断当前点与最近邻点的距离是否在NEAREST_FEATURE_SEARCH_SQ_DIST内，
        // 然后像LOAM一样筛选出3个最近邻点用于组织成一个平面
        if (pointSearchSqDis[0] < NEAREST_FEATURE_SEARCH_SQ_DIST) {
          closestPointInd = pointSearchInd[0];
          int closestPointScan =
              int(laserCloudSurfLast->points[closestPointInd].intensity);

          float pointSqDis, minPointSqDis2 = NEAREST_FEATURE_SEARCH_SQ_DIST,
                            minPointSqDis3 = NEAREST_FEATURE_SEARCH_SQ_DIST;

          for (int j = closestPointInd + 1; j < surfPointsFlatNum; j++) {
            if (int(laserCloudSurfLast->points[j].intensity) >
                closestPointScan + 2.5) {
              break;
            }

            pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                             (laserCloudSurfLast->points[j].x - pointSel.x) +
                         (laserCloudSurfLast->points[j].y - pointSel.y) *
                             (laserCloudSurfLast->points[j].y - pointSel.y) +
                         (laserCloudSurfLast->points[j].z - pointSel.z) *
                             (laserCloudSurfLast->points[j].z - pointSel.z);
            if (int(laserCloudSurfLast->points[j].intensity) <=
                closestPointScan) {
              if (pointSqDis < minPointSqDis2) {
                minPointSqDis2 = pointSqDis;
                minPointInd2 = j;
              }
            } else {
              if (pointSqDis < minPointSqDis3) {
                minPointSqDis3 = pointSqDis;
                minPointInd3 = j;
              }
            }
          }

          for (int j = closestPointInd - 1; j >= 0; j--) {
            if (int(laserCloudSurfLast->points[j].intensity) <
                closestPointScan - 2.5) {
              break;
            }

            pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                             (laserCloudSurfLast->points[j].x - pointSel.x) +
                         (laserCloudSurfLast->points[j].y - pointSel.y) *
                             (laserCloudSurfLast->points[j].y - pointSel.y) +
                         (laserCloudSurfLast->points[j].z - pointSel.z) *
                             (laserCloudSurfLast->points[j].z - pointSel.z);

            if (int(laserCloudSurfLast->points[j].intensity) >=
                closestPointScan) {
              if (pointSqDis < minPointSqDis2) {
                minPointSqDis2 = pointSqDis;
                minPointInd2 = j;
              }
            } else {
              if (pointSqDis < minPointSqDis3) {
                minPointSqDis3 = pointSqDis;
                minPointInd3 = j;
              }
            }
          }
        }
        // 3个最近邻点索引
        pointSearchSurfInd1[i] = closestPointInd;
        pointSearchSurfInd2[i] = minPointInd2;
        pointSearchSurfInd3[i] = minPointInd3;
      }

      if (pointSearchSurfInd2[i] >= 0 && pointSearchSurfInd3[i] >= 0) {
        // 3个最近邻点
        tripod1 = laserCloudSurfLast->points[pointSearchSurfInd1[i]];
        tripod2 = laserCloudSurfLast->points[pointSearchSurfInd2[i]];
        tripod3 = laserCloudSurfLast->points[pointSearchSurfInd3[i]];

        // 当前点
        V3D P0xyz(pointSel.x, pointSel.y, pointSel.z);
        // 3个最近邻点
        V3D P1xyz(tripod1.x, tripod1.y, tripod1.z);
        V3D P2xyz(tripod2.x, tripod2.y, tripod2.z);
        V3D P3xyz(tripod3.x, tripod3.y, tripod3.z);

        // M为假想平面的方向向量
        V3D M = math_utils::skew(P1xyz - P2xyz) * (P1xyz - P3xyz);
        double r = (P0xyz - P1xyz).transpose() * M;
        double m = M.norm();
        // 残差项，也就是点面距离 res = h(x) - z, 如果是面点距离,则是res = z - h(x)
        float res = r / m; // 注意这里没加绝对值,意思是有可能为负,这对后面的测量更新方程影响极大,不能加绝对值

        // 该特征点对应的雅克比矩阵
        // 点线距离分别对x0,y0,z0的偏导，最终要对transform的偏导数，这里就是平面的法向量？
        // 平面的法向量就是雅克比也说得通，比较沿该方向点面距离下降最快
        V3D jacxyz = M.transpose() / (m);

        float s = 1;
        //权重s计算，距离越大权重越小，距离越小权重越大，得到的权重范围<=1
        if (iterCount >= ICP_FREQ) {
          //增加权重，距离越远，影响影子越小
          s = 1 -
              1.8 * fabs(res) /
                  sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y * pointSel.y +
                            pointSel.z * pointSel.z));
        }

        // 满足权重阈值且残差项不为0，将特征点插入
        if (s > 0.1 && res != 0) {
          coeff.x = s * jacxyz(0);
          coeff.y = s * jacxyz(1);
          coeff.z = s * jacxyz(2);
          coeff.intensity = s * res;

          // 每个特征点对应的Jaccobian矩阵的三个元素都保存在coeff中,后面采用L-M方法解算的时候直接调用就行了
          keypoints->push_back(newScan->surfPointsFlat_->points[i]);
          jacobianCoff->push_back(coeff);
        }
      }
    }
  }

  void findCorrespondingCornerFeatures(
      ScanPtr lastScan, ScanPtr newScan,
      pcl::PointCloud<PointType>::Ptr keypoints,
      pcl::PointCloud<PointType>::Ptr jacobianCoff, int iterCount) {
    int cornerPointsSharpNum = newScan->cornerPointsSharp_->points.size();

    for (int i = 0; i < cornerPointsSharpNum; i++) {
      PointType pointSel;
      PointType coeff, tripod1, tripod2;

      transformToStart(&newScan->cornerPointsSharp_->points[i], &pointSel);

      pcl::PointCloud<PointType>::Ptr laserCloudCornerLast =
          lastScan->cornerPointsLessSharp_;

      if (iterCount % ICP_FREQ == 0) {
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        kdtreeCorner_->nearestKSearch(pointSel, 1, pointSearchInd,
                                      pointSearchSqDis);
        int closestPointInd = -1, minPointInd2 = -1;

        if (pointSearchSqDis[0] < NEAREST_FEATURE_SEARCH_SQ_DIST) {
          closestPointInd = pointSearchInd[0];
          int closestPointScan =
              int(laserCloudCornerLast->points[closestPointInd].intensity);

          float pointSqDis, minPointSqDis2 = NEAREST_FEATURE_SEARCH_SQ_DIST;
          for (int j = closestPointInd + 1; j < cornerPointsSharpNum; j++) {
            if (int(laserCloudCornerLast->points[j].intensity) >
                closestPointScan + 2.5) {
              break;
            }

            pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                             (laserCloudCornerLast->points[j].x - pointSel.x) +
                         (laserCloudCornerLast->points[j].y - pointSel.y) *
                             (laserCloudCornerLast->points[j].y - pointSel.y) +
                         (laserCloudCornerLast->points[j].z - pointSel.z) *
                             (laserCloudCornerLast->points[j].z - pointSel.z);

            if (int(laserCloudCornerLast->points[j].intensity) >
                closestPointScan) {
              if (pointSqDis < minPointSqDis2) {
                minPointSqDis2 = pointSqDis;
                minPointInd2 = j;
              }
            }
          }
          for (int j = closestPointInd - 1; j >= 0; j--) {
            if (int(laserCloudCornerLast->points[j].intensity) <
                closestPointScan - 2.5) {
              break;
            }

            pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                             (laserCloudCornerLast->points[j].x - pointSel.x) +
                         (laserCloudCornerLast->points[j].y - pointSel.y) *
                             (laserCloudCornerLast->points[j].y - pointSel.y) +
                         (laserCloudCornerLast->points[j].z - pointSel.z) *
                             (laserCloudCornerLast->points[j].z - pointSel.z);

            if (int(laserCloudCornerLast->points[j].intensity) <
                closestPointScan) {
              if (pointSqDis < minPointSqDis2) {
                minPointSqDis2 = pointSqDis;
                minPointInd2 = j;
              }
            }
          }
        }

        pointSearchCornerInd1[i] = closestPointInd;
        pointSearchCornerInd2[i] = minPointInd2;
      }

      if (pointSearchCornerInd2[i] >= 0) {
        tripod1 = laserCloudCornerLast->points[pointSearchCornerInd1[i]];
        tripod2 = laserCloudCornerLast->points[pointSearchCornerInd2[i]];

        V3D P0xyz(pointSel.x, pointSel.y, pointSel.z);
        V3D P1xyz(tripod1.x, tripod1.y, tripod1.z);
        V3D P2xyz(tripod2.x, tripod2.y, tripod2.z);

        V3D P = math_utils::skew(P0xyz - P1xyz) * (P0xyz - P2xyz);
        float r = P.norm();
        float d12 = (P1xyz - P2xyz).norm();
        float res = r / d12;

        V3D jacxyz =
            P.transpose() * math_utils::skew(P2xyz - P1xyz) / (d12 * r);

        float s = 1;
        if (iterCount >= ICP_FREQ) {
          s = 1 - 1.8 * fabs(res);
        }

        if (s > 0.1 && res != 0) {
          coeff.x = s * jacxyz(0);
          coeff.y = s * jacxyz(1);
          coeff.z = s * jacxyz(2);
          coeff.intensity = s * res;

          keypoints->push_back(newScan->cornerPointsSharp_->points[i]);
          jacobianCoff->push_back(coeff);
        }
      }
    }
  }

  // Undistort point cloud to the start frame
  void transformToStart(PointType const* const pi, PointType* const po) {
    double s = (1.f / SCAN_PERIOD) * (pi->intensity - int(pi->intensity));

    V3D P2xyz(pi->x, pi->y, pi->z);
    V3D phi = Quat2axis(linState_.qbn_);
    Q4D R21xyz = axis2Quat(s * phi);
    R21xyz.normalized();
    V3D T112xyz = s * linState_.rn_;
    V3D P1xyz = R21xyz * P2xyz + T112xyz;

    po->x = P1xyz.x();
    po->y = P1xyz.y();
    po->z = P1xyz.z();
    po->intensity = pi->intensity;
  }

  // Undistort point cloud to the end frame
  void transformToEnd(PointType const* const pi, PointType* const po) {
    double s = (1.f / SCAN_PERIOD) * (pi->intensity - int(pi->intensity));

    V3D P2xyz(pi->x, pi->y, pi->z);
    V3D phi = Quat2axis(linState_.qbn_);
    Q4D R21xyz = axis2Quat(s * phi);
    R21xyz.normalized();
    V3D T112xyz = s * linState_.rn_;
    V3D P1xyz = R21xyz * P2xyz + T112xyz;

    R21xyz = linState_.qbn_;
    T112xyz = linState_.rn_;
    P2xyz = R21xyz.inverse() * (P1xyz - T112xyz);

    po->x = P2xyz.x();
    po->y = P2xyz.y();
    po->z = P2xyz.z();
    po->intensity = pi->intensity;
  }

  // Coordinate transformation from LiDAR frame to Vehicle frame
  // lidar frame --> vehicle frame
  void rotatePoint(PointType const* const pi, PointType* const po) {
    V3D rpy;
    // IMU_LIDAR_EXTRINSIC_ANGLE为imu和lidar外参中的yaw，参数文件中设置了为0
    rpy << deg2rad(0.0), deg2rad(0.0), deg2rad(IMU_LIDAR_EXTRINSIC_ANGLE);
    M3D R = rpy2R(rpy);
    V3D Pi(pi->x, pi->y, pi->z);
    V3D Po = R * Pi;
    po->x = Po.x();
    po->y = Po.y();
    po->z = Po.z();
    po->intensity = pi->intensity;
  }

  void updatePointCloud() {
    scan_new_->cornerPointsLessSharpYZX_->clear();
    scan_new_->surfPointsLessFlatYZX_->clear();
    scan_new_->outlierPointCloudYZX_->clear();

    PointType point;
    for (int i = 0; i < scan_new_->cornerPointsLessSharp_->points.size(); i++) {
      transformToEnd(&scan_new_->cornerPointsLessSharp_->points[i],
                     &scan_new_->cornerPointsLessSharp_->points[i]);
      point.x = scan_new_->cornerPointsLessSharp_->points[i].y;
      point.y = scan_new_->cornerPointsLessSharp_->points[i].z;
      point.z = scan_new_->cornerPointsLessSharp_->points[i].x;
      point.intensity = scan_new_->cornerPointsLessSharp_->points[i].intensity;
      scan_new_->cornerPointsLessSharpYZX_->push_back(point);
    }
    for (int i = 0; i < scan_new_->surfPointsLessFlat_->points.size(); i++) {
      transformToEnd(&scan_new_->surfPointsLessFlat_->points[i],
                     &scan_new_->surfPointsLessFlat_->points[i]);
      point.x = scan_new_->surfPointsLessFlat_->points[i].y;
      point.y = scan_new_->surfPointsLessFlat_->points[i].z;
      point.z = scan_new_->surfPointsLessFlat_->points[i].x;
      point.intensity = scan_new_->surfPointsLessFlat_->points[i].intensity;
      scan_new_->surfPointsLessFlatYZX_->push_back(point);
    }
    for (int i = 0; i < scan_new_->outlierPointCloud_->points.size(); i++) {
      // transformToEnd(&scan_new_->outlierPointCloud_->points[i],
      //                &scan_new_->outlierPointCloud_->points[i]);
      point.x = scan_new_->outlierPointCloud_->points[i].y;
      point.y = scan_new_->outlierPointCloud_->points[i].z;
      point.z = scan_new_->outlierPointCloud_->points[i].x;
      point.intensity = scan_new_->outlierPointCloud_->points[i].intensity;
      scan_new_->outlierPointCloudYZX_->push_back(point);
    }

    // Transform XYZ-convention to YZX-convention to meet the mapping module's
    // requirement
    globalStateYZX_.rn_ = Q_xyz_to_yzx * globalState_.rn_;
    globalStateYZX_.qbn_ =
        Q_xyz_to_yzx * globalState_.qbn_ * Q_xyz_to_yzx.inverse();

    if (scan_new_->cornerPointsLessSharp_->points.size() >= 5 &&
        scan_new_->surfPointsLessFlat_->points.size() >= 20) {
      kdtreeCorner_->setInputCloud(scan_new_->cornerPointsLessSharp_);
      kdtreeSurf_->setInputCloud(scan_new_->surfPointsLessFlat_);
    }
  }

  void estimateTransform(ScanPtr lastScan, ScanPtr newScan, V3D& t, Q4D& q) {
    // 预积分的时间增量
    double sum_dt = preintegration_->sum_dt;
    // 存储imu预积分预测得到的帧间位姿变换，这里以它为icp迭代初值可以加速迭代速度
    linState_.rn_ = t;
    linState_.qbn_ = q;
    // NUM_ITER默认为30，ICP
    for (int iter = 0; iter < NUM_ITER; iter++) {
      keypointSurfs_->clear();
      jacobianCoffSurfs->clear();
      keypointCorns_->clear();
      jacobianCoffCorns->clear();

      // 寻找平面特征点及其对应雅克比
      findCorrespondingSurfFeatures(lastScan, newScan, keypointSurfs_,
                                    jacobianCoffSurfs, iter);
      // 关键点数目要足够才可信
      if (keypointSurfs_->points.size() < 10) {
        ROS_WARN("Insufficient matched surfs...");
        continue;
      }
      // 寻找边缘特征点及其对应雅克比
      findCorrespondingCornerFeatures(lastScan, newScan, keypointCorns_,
                                      jacobianCoffCorns, iter);
      // 同上
      if (keypointCorns_->points.size() < 5) {
        ROS_WARN("Insufficient matched corners...");
        continue;
      }

      // ICP求帧间位姿变换
      if (calculateTransformation(lastScan, newScan, keypointCorns_,
                                  jacobianCoffCorns, keypointSurfs_,
                                  jacobianCoffSurfs, iter)) {
        ROS_INFO_STREAM("System Converges after " << iter << " iterations");
        break;
      }
    }

    // 更新帧间位姿变换
    t = linState_.rn_;
    q = linState_.qbn_;  // qbn_ is quaternion rotation from b-frame to n-frame
  }

  bool calculateTransformation(ScanPtr lastScan, ScanPtr newScan,
                               pcl::PointCloud<PointType>::Ptr corners,
                               pcl::PointCloud<PointType>::Ptr jacoCornersCoff,
                               pcl::PointCloud<PointType>::Ptr surfs,
                               pcl::PointCloud<PointType>::Ptr jacoSurfsCoff,
                               int iterCount) {
    keypoints_->clear();
    jacobians_->clear();
    (*keypoints_) += (*surfs);
    (*keypoints_) += (*corners);
    (*jacobians_) += (*jacoSurfsCoff);
    (*jacobians_) += (*jacoCornersCoff);

    const int stateNum = 6;
    const int pointNum = keypoints_->points.size();
    const int imuNum = 0;
    const int row = pointNum + imuNum;
    Eigen::Matrix<double, Eigen::Dynamic, stateNum> J(row, stateNum);
    Eigen::Matrix<double, stateNum, Eigen::Dynamic> JT(stateNum, row);
    Eigen::Matrix<double, stateNum, stateNum> JTJ;
    Eigen::VectorXd b(row);
    Eigen::Matrix<double, stateNum, 1> JTb;
    Eigen::Matrix<double, stateNum, 1> x;
    J.setZero();
    JT.setZero();
    JTJ.setZero();
    b.setZero();
    JTb.setZero();
    x.setZero();

    // 组织雅克比矩阵J和齐次项b
    for (int i = 0; i < pointNum; ++i) {
      // Select keypoint i
      const PointType& keypoint = keypoints_->points[i];
      const PointType& coeff = jacobians_->points[i];

      V3D P2xyz(keypoint.x, keypoint.y, keypoint.z);
      V3D coff_xyz(coeff.x, coeff.y, coeff.z);

      double s =
          (1.f / SCAN_PERIOD) * (keypoint.intensity - int(keypoint.intensity));

      V3D phi = Quat2axis(linState_.qbn_);
      // Rotation matrix from frame2 (new) to frame1 (last)
      Q4D R21xyz = axis2Quat(s * phi);
      R21xyz.normalized();
      // Translation vector from frame1 to frame2 represented in frame1
      V3D T112xyz = s * linState_.rn_;

      V3D jacobian1xyz =
          coff_xyz.transpose() *
          (-R21xyz.toRotationMatrix() * skew(P2xyz));  // rotation jacobian
      V3D jacobian2xyz =
          coff_xyz.transpose() * M3D::Identity();  // translation jacobian
      double residual = coeff.intensity;

      J.block<1, 3>(i, O_R) = jacobian1xyz;
      J.block<1, 3>(i, O_P) = jacobian2xyz;

      // Set the overall residual
      b(i) = -0.05 * residual;
    }

    // L-M迭代大法
    // Solve x
    JT = J.transpose();
    JTJ = JT * J;
    JTb = JT * b;
    // L-M 的增量delta x
    x = JTJ.colPivHouseholderQr().solve(JTb);

    // Determine whether x is degenerated
    // 需要判断x是否退化
    bool isDegenerate = false;
    Eigen::Matrix<double, stateNum, stateNum> matP;
    if (iterCount == 0) {
      Eigen::Matrix<double, 1, stateNum> matE;
      Eigen::Matrix<double, stateNum, stateNum> matV;
      Eigen::Matrix<double, stateNum, stateNum> matV2;

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, stateNum, stateNum> >
          esolver(JTJ);
      matE = esolver.eigenvalues().real();
      matV = esolver.eigenvectors().real();

      matV2 = matV;

      isDegenerate = false;
      std::vector<double> eignThre(stateNum, 10.);
      for (int i = 0; i < stateNum; i++) {
        // if eigenvalue is less than 10, set the corresponding eigenvector to 0
        // vector
        if (matE(0, i) < eignThre[i]) {
          for (int j = 0; j < stateNum; j++) {
            matV2(i, j) = 0;
          }
          isDegenerate = true;
        } else {
          break;
        }
      }
      matP = matV.inverse() * matV2;
    }

    // 判断退化
    if (isDegenerate) {
      cout << "System is Degenerate." << endl;
      Eigen::Matrix<double, stateNum, 1> matX2(x);
      x = matP * matX2;
    }

    // Update state linState_
    Q4D dq = rpy2Quat(x.segment<3>(O_R));
    linState_.qbn_ = (linState_.qbn_ * dq).normalized();
    linState_.rn_ += x.segment<3>(O_P);

    // Determine whether should it stop
    V3D rpy_rad = x.segment<3>(O_R);
    V3D rpy_deg = math_utils::rad2deg(rpy_rad);
    double deltaR = rpy_deg.norm();
    V3D trans = 100 * x.segment<3>(O_P);
    double deltaT = trans.norm();
    // 增量很小了，那么意味着精度足够高，优化成功
    if (deltaR < 0.1 && deltaT < 0.1) {
      return true;
    }

    return false;
  }

  void estimateInitialState1(const V3D& p, const Q4D& q, V3D& v0, V3D& v1,
                             V3D& ba, V3D& bw) {
    ba = INIT_BA;
    bw = INIT_BW;

    solveGyroscopeBias(q, bw);

    double sum_dt = preintegration_->sum_dt;
    v0 =
        (p - 0.5 * linState_.gn_ * sum_dt * sum_dt - preintegration_->delta_p) /
        sum_dt;
    v1 = v0 + sum_dt * linState_.gn_ + preintegration_->delta_v;

    cout << "v0: " << v0.transpose() << endl;
    cout << "v1: " << v1.transpose() << endl;
    cout << "ba0: " << INIT_BA.transpose() << endl;
    cout << "bw0: " << INIT_BW.transpose() << endl;
    cout << "bw0: " << bw.transpose() << endl;
  }

  void estimateInitialState2(const V3D& p, const Q4D& q, V3D& v0, V3D& v1,
                             V3D& ba, V3D& bw) {
    const int DIM_OF_STATE = 1 + 1 + 3;
    const int DIM_OF_MEAS = 3 + 3;
    Eigen::Matrix<double, Eigen::Dynamic, DIM_OF_STATE> J(DIM_OF_MEAS,
                                                          DIM_OF_STATE);
    Eigen::Matrix<double, DIM_OF_STATE, Eigen::Dynamic> JT(DIM_OF_STATE,
                                                           DIM_OF_MEAS);
    Eigen::Matrix<double, DIM_OF_STATE, DIM_OF_STATE> JTJ;
    Eigen::VectorXd b(DIM_OF_MEAS);
    Eigen::Matrix<double, DIM_OF_STATE, 1> JTb;
    Eigen::Matrix<double, DIM_OF_STATE, 1> x;

    J.setZero();
    JT.setZero();
    JTJ.setZero();
    b.setZero();
    JTb.setZero();
    x.setZero();

    double sum_dt = preintegration_->sum_dt;
    b.block<3, 1>(0, 0) =
        (p - preintegration_->delta_p) / sum_dt - 0.5 * sum_dt * linState_.gn_;
    b.block<3, 1>(3, 0) = sum_dt * linState_.gn_ + preintegration_->delta_v;

    V3D L(1, 0, 0);
    J.block<3, 1>(0, 0) = L;
    J.block<3, 3>(0, 2) = -0.5 * sum_dt * M3D::Identity();
    J.block<3, 1>(3, 0) = -L;
    J.block<3, 1>(3, 1) = L;
    J.block<3, 3>(3, 2) = sum_dt * M3D::Identity();

    JT = J.transpose();
    JTJ = JT * J;
    JTb = JT * b;

    x = JTJ.colPivHouseholderQr().solve(JTb);
    v0 = x(0) * L;
    v1 = x(1) * L;
    ba = INIT_BA;
    bw = INIT_BW;

    V3D test_ba = linState_.gn_ + preintegration_->delta_v / sum_dt;
    cout << "test_ba: " << test_ba.transpose() << endl;
  }

  void estimateInitialState3(const V3D& p, const Q4D& q, V3D& v0, V3D& v1,
                             V3D& ba, V3D& bw) {
    double sum_dt = preintegration_->sum_dt;
    V3D v = p / sum_dt;
    // v * sum_dt = (p - 0.5*linState_.gn_*sum_dt*sum_dt -
    // preintegration_->delta_p + 0.5*ba*sum_dt*sum_dt);
    ba = (v * sum_dt - p + 0.5 * linState_.gn_ * sum_dt * sum_dt +
          preintegration_->delta_p) *
         2 * (1.0 / sum_dt * sum_dt);

    solveGyroscopeBias(q, bw);

    v0 = v;
    v1 = v;
    cout << "v0: " << v0.transpose() << endl;
    cout << "v1: " << v1.transpose() << endl;
    cout << "ba0: " << ba.transpose() << endl;
    cout << "bw0: " << bw.transpose() << endl;
  }

  // 估计初始状态
  void estimateInitialState(const V3D& p, const Q4D& q, V3D& v0, V3D& v1,
                            V3D& ba, V3D& bw) {
    // 帧间时间
    double sum_dt = preintegration_->sum_dt;
    // Calculate a rough velocity using relative translation
    // 认为是匀速运动
    V3D v = p / sum_dt;
    // start velocity?
    v0 = v;
    // end velocity?
    v1 = v;
    // TODO(charles): calibrate initial ba and bw using two consecutive scans
    // and IMU preintegration results
    ba = INIT_BA;
    bw = INIT_BW;
  }

  // Estimate gyroscope bias using a similar methoed provided in VINS-Mono
  // 估计角速度误差，具体要参考VINS
  void solveGyroscopeBias(const Q4D& q, V3D& bw) {
    Matrix3d A;
    V3D b;
    V3D delta_bg;
    A.setZero();
    b.setZero();

    MatrixXd tmp_A(3, 3);
    tmp_A.setZero();
    VectorXd tmp_b(3);
    tmp_b.setZero();
    Eigen::Quaterniond q_ij = q;
    tmp_A = preintegration_->jacobian.template block<3, 3>(GlobalState::att_,
                                                           GlobalState::gyr_);
    tmp_b = 2 * (preintegration_->delta_q.inverse() * q_ij).vec();
    A += tmp_A.transpose() * tmp_A;
    b += tmp_A.transpose() * tmp_b;

    delta_bg = A.ldlt().solve(b);
    ROS_WARN_STREAM("gyroscope bias initial calibration "
                    << delta_bg.transpose());

    bw += delta_bg;
  }

 public:
  FusionStatus status_;     // system status
  StatePredictor* filter_;  // Kalman filter pointer
  ScanPtr scan_new_;        // current scan information
  ScanPtr scan_last_;       // last scan information

  // !@KD tree relatives
  pcl::VoxelGrid<PointType> downSizeFilter_;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeCorner_;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurf_;

  // !@Feature matching relatives
  std::vector<int> pointSelCornerInd;
  std::vector<double> pointSearchCornerInd1;
  std::vector<double> pointSearchCornerInd2;
  std::vector<int> pointSelSurfInd;
  std::vector<double> pointSearchSurfInd1;
  std::vector<double> pointSearchSurfInd2;
  std::vector<double> pointSearchSurfInd3;

  // !@Jacobians and keypoints
  pcl::PointCloud<PointType>::Ptr keypoints_;
  pcl::PointCloud<PointType>::Ptr jacobians_;
  pcl::PointCloud<PointType>::Ptr keypointCorns_;
  pcl::PointCloud<PointType>::Ptr keypointSurfs_;
  pcl::PointCloud<PointType>::Ptr jacobianCoffCorns;
  pcl::PointCloud<PointType>::Ptr jacobianCoffSurfs;

  // !@Global transformation from the original scan-frame to current scan-frame
  GlobalState globalState_;
  // !@Relative transformation from scan0-frame t0 scan1-frame
  GlobalState linState_;
  Eigen::Matrix<double, GlobalState::DIM_OF_STATE_, 1> difVecLinInv_;
  Eigen::Matrix<double, GlobalState::DIM_OF_STATE_, 1> updateVec_;
  double updateVecNorm_ = 0.0;

  // !@Kalman filter relatives
  VXD residual_;
  MXD Fk_;
  MXD Gk_;
  MXD Pk_;
  MXD Qk_;
  MXD Rk_;
  MXD Hk_;
  MXD Jk_;
  MXD Kk_;
  MXD IKH_;
  MXD Py_;
  MXD Pyinv_;

  // !@ IMU preintegration
  integration::IntegrationBase* preintegration_;
  Imu imu_last_;

  // !@Rotation matrices between XYZ-convention and YZX-convention
  Eigen::Matrix3d R_yzx_to_xyz;
  Eigen::Matrix3d R_xyz_to_yzx;
  Eigen::Quaterniond Q_yzx_to_xyz;
  Eigen::Quaterniond Q_xyz_to_yzx;
  GlobalState globalStateYZX_;
};

}  // namespace fusion

#endif  // INCLUDE_STATEESTIMATOR_HPP_
