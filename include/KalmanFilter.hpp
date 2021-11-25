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

#ifndef INCLUDE_KALMANFILTER_HPP_
#define INCLUDE_KALMANFILTER_HPP_

#include <math_utils.h>
#include <parameters.h>

#include <iostream>
#include <map>

using namespace std;
using namespace math_utils;
using namespace parameter;

namespace filter {

// GlobalState Class contains state variables including position, velocity,
// attitude, acceleration bias, gyroscope bias, and gravity
// 该类包含位置，速度，姿态，加速度计误差，陀螺仪误差和重力，论文公式(2)
class GlobalState {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // 状态向量维度
  static constexpr unsigned int DIM_OF_STATE_ = 18;
  // 噪声向量维度
  static constexpr unsigned int DIM_OF_NOISE_ = 12;
  // 各变量的首元素索引值
  static constexpr unsigned int pos_ = 0;
  static constexpr unsigned int vel_ = 3;
  static constexpr unsigned int att_ = 6;
  static constexpr unsigned int acc_ = 9;
  static constexpr unsigned int gyr_ = 12;
  static constexpr unsigned int gra_ = 15;

  GlobalState() { setIdentity(); }

  // 设置状态量
  GlobalState(const V3D& rn, const V3D& vn, const Q4D& qbn, const V3D& ba,
              const V3D& bw) {
    setIdentity();
    rn_ = rn; // position
    vn_ = vn; // velocity
    qbn_ = qbn; // attitude
    ba_ = ba; // acceleration bias
    bw_ = bw; // gyroscope bias
  }

  // 析构，删除当前对象
  ~GlobalState() {}

  // 初始化时使用
  void setIdentity() {
    rn_.setZero();
    vn_.setZero();
    qbn_.setIdentity();
    ba_.setZero();
    bw_.setZero();
    gn_ << 0.0, 0.0, -G0; // gravity, 9.81
  }

  // boxPlus operator
  // 定义boxplus operator, 论文公式(19)
  void boxPlus(const Eigen::Matrix<double, DIM_OF_STATE_, 1>& xk,
               GlobalState& stateOut) {
    stateOut.rn_ = rn_ + xk.template segment<3>(pos_);
    stateOut.vn_ = vn_ + xk.template segment<3>(vel_);
    stateOut.ba_ = ba_ + xk.template segment<3>(acc_);
    stateOut.bw_ = bw_ + xk.template segment<3>(gyr_);
    // 欧拉角转四元数
    Q4D dq = axis2Quat(xk.template segment<3>(att_));
    stateOut.qbn_ = (qbn_ * dq).normalized(); 

    stateOut.gn_ = gn_ + xk.template segment<3>(gra_);
  }

  // boxMinus operator
  // 定义boxMinus operator, 类似论文公式(19)
  void boxMinus(const GlobalState& stateIn,
                Eigen::Matrix<double, DIM_OF_STATE_, 1>& xk) {
    xk.template segment<3>(pos_) = rn_ - stateIn.rn_;
    xk.template segment<3>(vel_) = vn_ - stateIn.vn_;
    xk.template segment<3>(acc_) = ba_ - stateIn.ba_;
    xk.template segment<3>(gyr_) = bw_ - stateIn.bw_;
    V3D da = Quat2axis(stateIn.qbn_.inverse() * qbn_);
    xk.template segment<3>(att_) = da;

    xk.template segment<3>(gra_) = gn_ - stateIn.gn_;
  }

  // 重载"="
  GlobalState& operator=(const GlobalState& other) {
    if (this == &other) return *this;

    this->rn_ = other.rn_;
    this->vn_ = other.vn_;
    this->qbn_ = other.qbn_;
    this->ba_ = other.ba_;
    this->bw_ = other.bw_;
    this->gn_ = other.gn_;

    return *this;
  }

  // !@State
  // 各状态变量，共18维的向量
  V3D rn_;   // position in n-frame
  V3D vn_;   // velocity in n-frame
  Q4D qbn_;  // rotation from b-frame to n-frame
  V3D ba_;   // acceleartion bias
  V3D bw_;   // gyroscope bias
  V3D gn_;   // gravity
};

// 状态预测器
class StatePredictor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  StatePredictor() { reset(); }

  ~StatePredictor() {}

  bool predict(double dt, const V3D& acc, const V3D& gyr,
               bool update_jacobian_ = true) {
    // 系统未初始化
    if (!isInitialized()) return false;

    // 未初始化Imu
    if (!flag_init_imu_) {
      flag_init_imu_ = true;
      acc_last = acc;
      gyr_last = gyr;
    }

    // Average acceleration and angular rate 滤波作用？
    // 这里的状态不是误差状态,是正儿八经的状态[pos,vel,att,acc,gyr,gra]
    GlobalState state_tmp = state_;
    // 去除上一次加速度中的重力成分,乘以姿态即转到世界坐标系下
    V3D un_acc_0 = state_tmp.qbn_ * (acc_last - state_tmp.ba_) + state_tmp.gn_;
    // 与上一次角速度做加权平均
    V3D un_gyr = 0.5 * (gyr_last + gyr) - state_tmp.bw_;
    // 求角度变化量，再转化成四元数形式
    Q4D dq = axis2Quat(un_gyr * dt);
    // 求当前临时状态量中的姿态
    state_tmp.qbn_ = (state_tmp.qbn_ * dq).normalized();
    // 去除本次加速度中的重力成分
    V3D un_acc_1 = state_tmp.qbn_ * (acc - state_tmp.ba_) + state_tmp.gn_;
    // 两次加速度加权平均
    V3D un_acc = 0.5 * (un_acc_0 + un_acc_1);

    // State integral
    // 通过IMU测量进行积分，得到新的position和velocity
    // x = x + v*t + a*t^2
    // // v = v + a*t
    state_tmp.rn_ = state_tmp.rn_ + dt * state_tmp.vn_ + 0.5 * dt * dt * un_acc;
    state_tmp.vn_ = state_tmp.vn_ + dt * un_acc;

    // 更新jacobian矩阵
    if (update_jacobian_) {
      // 论文公式(6),注意这里的状态会比论文中多了重力加速度的3d向量
      MXD Ft =
          MXD::Zero(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_); // 18*18
      Ft.block<3, 3>(GlobalState::pos_, GlobalState::vel_) = M3D::Identity();

      Ft.block<3, 3>(GlobalState::vel_, GlobalState::att_) =
          -state_tmp.qbn_.toRotationMatrix() * skew(acc - state_tmp.ba_);
      Ft.block<3, 3>(GlobalState::vel_, GlobalState::acc_) =
          -state_tmp.qbn_.toRotationMatrix();
      Ft.block<3, 3>(GlobalState::vel_, GlobalState::gra_) = M3D::Identity();

      Ft.block<3, 3>(GlobalState::att_, GlobalState::att_) =
          skew(gyr - state_tmp.bw_);
      Ft.block<3, 3>(GlobalState::att_, GlobalState::gyr_) = -M3D::Identity();

      // 论文公式(7),注意这里的状态会比论文中多了重力加速度的3d向量
      MXD Gt =
          MXD::Zero(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_NOISE_); // 18*12
      Gt.block<3, 3>(GlobalState::vel_, 0) = -state_tmp.qbn_.toRotationMatrix();
      Gt.block<3, 3>(GlobalState::att_, 3) = -M3D::Identity();
      Gt.block<3, 3>(GlobalState::acc_, 6) = M3D::Identity();
      Gt.block<3, 3>(GlobalState::gyr_, 9) = M3D::Identity();
      Gt = Gt * dt;

      // 这里应该对应论文公式(9)更新预测协方差，但是公式好像不太对应
      // 解答:论文中的公式是通过泰勒展开后去一阶项,因此有F_ = I + Ft * dt,这里为了提高
      // 求解精度,不妨取到二阶项,因为Ft比较稀疏,计算耗时不高
      // 具体参考《Estimation techniques for low-cost inertial navigation》
      const MXD I =
          MXD::Identity(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_);
      F_ = I + Ft * dt + 0.5 * Ft * Ft * dt * dt;

      // jacobian_ = F * jacobian_;
      covariance_ =
          F_ * covariance_ * F_.transpose() + Gt * noise_ * Gt.transpose();
      // 这样可近似原始协方差矩阵,但是它是半正定的,会更鲁棒
      covariance_ = 0.5 * (covariance_ + covariance_.transpose()).eval();
    }

    // 更新滤波器当前状态
    state_ = state_tmp;
    // 更新滤波器当前时间戳
    time_ += dt;
    // 记录本次IMU测量，下次会使用
    acc_last = acc;
    gyr_last = gyr;
    return true;
  }

  // 从imu测量计算当前的roll和pitch
  static void calculateRPfromIMU(const V3D& acc, double& roll, double& pitch) {
    pitch = -sign(acc.z()) * asin(acc.x() / G0);
    roll = sign(acc.z()) * asin(acc.y() / G0);
  }

  // 设置滤波器状态
  void set(const GlobalState& state) { state_ = state; }

  // 更新滤波器的状态及协方差
  void update(const GlobalState& state,
              const Eigen::Matrix<double, GlobalState::DIM_OF_STATE_,
                                  GlobalState::DIM_OF_STATE_>& covariance) {
    state_ = state;
    covariance_ = covariance;
  }

  // 初始化滤波器
  void initialization(double time, const V3D& rn, const V3D& vn, const Q4D& qbn,
                      const V3D& ba, const V3D& bw) {
    state_ = GlobalState(rn, vn, qbn, ba, bw);
    time_ = time;
    flag_init_state_ = true;

    initializeCovariance();
  }

  void initialization(double time, const V3D& rn, const V3D& vn, const Q4D& qbn,
                      const V3D& ba, const V3D& bw, const V3D& acc,
                      const V3D& gyr) {
    state_ = GlobalState(rn, vn, qbn, ba, bw);
    time_ = time;
    acc_last = acc;
    gyr_last = gyr;
    flag_init_imu_ = true;
    flag_init_state_ = true;

    initializeCovariance();
  }

  void initialization(double time, const V3D& rn, const V3D& vn, const V3D& ba,
                      const V3D& bw, double roll = 0.0, double pitch = 0.0,
                      double yaw = 0.0) {
    state_ = GlobalState(rn, vn, rpy2Quat(V3D(roll, pitch, yaw)), ba, bw);
    time_ = time;
    flag_init_state_ = true;

    initializeCovariance();
  }

  void initialization(double time, const V3D& rn, const V3D& vn, const V3D& ba,
                      const V3D& bw, const V3D& acc, const V3D& gyr,
                      double roll = 0.0, double pitch = 0.0, double yaw = 0.0) {
    state_ = GlobalState(rn, vn, rpy2Quat(V3D(roll, pitch, yaw)), ba, bw);
    time_ = time;
    acc_last = acc;
    gyr_last = gyr;
    flag_init_imu_ = true;
    flag_init_state_ = true;

    initializeCovariance();
  }

  // 初始化协方差矩阵
  void initializeCovariance(int type = 0) {
    // 对角线上的矩阵元素
    double covX = pow(INIT_POS_STD(0), 2);
    double covY = pow(INIT_POS_STD(1), 2);
    double covZ = pow(INIT_POS_STD(2), 2);
    double covVx = pow(INIT_VEL_STD(0), 2);
    double covVy = pow(INIT_VEL_STD(1), 2);
    double covVz = pow(INIT_VEL_STD(2), 2);
    double covRoll = pow(deg2rad(INIT_ATT_STD(0)), 2);
    double covPitch = pow(deg2rad(INIT_ATT_STD(1)), 2);
    double covYaw = pow(deg2rad(INIT_ATT_STD(2)), 2);

    // 将元素平方
    V3D covPos = INIT_POS_STD.array().square();
    V3D covVel = INIT_VEL_STD.array().square();
    V3D covAcc = INIT_ACC_STD.array().square();
    V3D covGyr = INIT_GYR_STD.array().square();

    // ug: micro-gravity force -- 9.81/(10^6)
    double peba = pow(ACC_N * ug, 2);
    double pebg = pow(GYR_N * dph, 2);
    double pweba = pow(ACC_W * ugpsHz, 2);
    double pwebg = pow(GYR_W * dpsh, 2);
    V3D gra_cov(0.01, 0.01, 0.01);

    if (type == 0) {
      // Initialize using offline parameters
      covariance_.setZero();
      covariance_.block<3, 3>(GlobalState::pos_, GlobalState::pos_) =
          covPos.asDiagonal();  // pos
      covariance_.block<3, 3>(GlobalState::vel_, GlobalState::vel_) =
          covVel.asDiagonal();  // vel
      covariance_.block<3, 3>(GlobalState::att_, GlobalState::att_) =
          V3D(covRoll, covPitch, covYaw).asDiagonal();  // att
      covariance_.block<3, 3>(GlobalState::acc_, GlobalState::acc_) =
          covAcc.asDiagonal();  // ba
      covariance_.block<3, 3>(GlobalState::gyr_, GlobalState::gyr_) =
          covGyr.asDiagonal();  // bg
      covariance_.block<3, 3>(GlobalState::gra_, GlobalState::gra_) =
          gra_cov.asDiagonal();  // gravity
    } else if (type == 1) {
      // Inheritage previous covariance
      M3D vel_cov =
          covariance_.block<3, 3>(GlobalState::vel_, GlobalState::vel_);
      M3D acc_cov =
          covariance_.block<3, 3>(GlobalState::acc_, GlobalState::acc_);
      M3D gyr_cov =
          covariance_.block<3, 3>(GlobalState::gyr_, GlobalState::gyr_);
      M3D gra_cov =
          covariance_.block<3, 3>(GlobalState::gra_, GlobalState::gra_);

      covariance_.setZero();
      covariance_.block<3, 3>(GlobalState::pos_, GlobalState::pos_) =
          covPos.asDiagonal();  // pos
      covariance_.block<3, 3>(GlobalState::vel_, GlobalState::vel_) =
          vel_cov;  // vel
      covariance_.block<3, 3>(GlobalState::att_, GlobalState::att_) =
          V3D(covRoll, covPitch, covYaw).asDiagonal();  // att
      covariance_.block<3, 3>(GlobalState::acc_, GlobalState::acc_) = acc_cov;
      covariance_.block<3, 3>(GlobalState::gyr_, GlobalState::gyr_) = gyr_cov;
      covariance_.block<3, 3>(GlobalState::gra_, GlobalState::gra_) = gra_cov;
    }

    // 噪声?
    noise_.setZero(); // 12*12
    // asDiagonal()指将向量作为对角线构建对角矩阵
    noise_.block<3, 3>(0, 0) = V3D(peba, peba, peba).asDiagonal();
    noise_.block<3, 3>(3, 3) = V3D(pebg, pebg, pebg).asDiagonal();
    noise_.block<3, 3>(6, 6) = V3D(pweba, pweba, pweba).asDiagonal();
    noise_.block<3, 3>(9, 9) = V3D(pwebg, pwebg, pwebg).asDiagonal();
  }

  // 重置滤波器状态
  void reset(int type = 0) {
    if (type == 0) {
      state_.rn_.setZero();
      state_.vn_ = state_.qbn_.inverse() * state_.vn_;
      state_.qbn_.setIdentity();
      initializeCovariance();
    } else if (type == 1) {
      state_.rn_.setZero();
      state_.vn_ = state_.qbn_.inverse() * state_.vn_;
      state_.qbn_.setIdentity();
      state_.gn_ = state_.qbn_.inverse() * state_.gn_;
      state_.gn_ = state_.gn_ * 9.81 / state_.gn_.norm();
      initializeCovariance(1);
    }
  }

  void reset(V3D vn, V3D ba, V3D bw) {
    state_.setIdentity();
    state_.vn_ = vn;
    state_.ba_ = ba;
    state_.bw_ = bw;
    initializeCovariance();
  }

  // 返回初始化标志位
  inline bool isInitialized() { return flag_init_state_; }

  GlobalState state_;
  double time_;
  Eigen::Matrix<double, GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_>
      F_;
  Eigen::Matrix<double, GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_>
      jacobian_, covariance_;
  Eigen::Matrix<double, GlobalState::DIM_OF_NOISE_, GlobalState::DIM_OF_NOISE_>
      noise_;

  V3D acc_last;  // last acceleration measurement
  V3D gyr_last;  // last gyroscope measurement

  bool flag_init_state_;
  bool flag_init_imu_;
};

};  // namespace filter

#endif  // INCLUDE_KALMANFILTER_HPP_
