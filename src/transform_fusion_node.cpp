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

// #include "utility.h"
#include <parameters.h>
#include <sensor_msgs/NavSatStatus.h>
#include <sensor_msgs/NavSatFix.h>
#include "gps_common/conversions.h"
// #include "sleipnir_msgs/gps4control.h"
#include <sleipnir_msgs/sensorgps.h>

using namespace parameter;
using namespace std;
using namespace gps_common;
using namespace Eigen;
using namespace GeographicLib;

class TransformFusion {
 private:
  ros::NodeHandle nh;

  ros::Publisher pubLaserOdometry2;
  ros::Subscriber subLaserOdometry;
  ros::Subscriber subOdomAftMapped;
  ros::Publisher pubUTMOdom;
  ros::Publisher pubGPSforCtrl;

  nav_msgs::Odometry laserOdometry2;
  tf::StampedTransform laserOdometryTrans2;
  tf::TransformBroadcaster tfBroadcaster2;

  tf::StampedTransform map_2_camera_init_Trans;
  tf::TransformBroadcaster tfBroadcasterMap2CameraInit;

  tf::StampedTransform camera_2_vehicle_Trans;
  tf::TransformBroadcaster tfBroadcasterCamera2Baselink;

  float transformSum[6];
  float transformIncre[6];
  float transformMapped[6];
  float transformBefMapped[6];
  float transformAftMapped[6];

//   GPS measurement origination in map
//   double OriLon = 0.0;
//   double OriLat = 0.0;
//   double OriAlt = 0.0;
//   double OriYaw = 0.0;
//   double OriPitch = 0.0;
//   double OriRoll = 0.0;
//   double compensate_init_yaw, compensate_init_pitch, compensate_init_roll;
//   double mappingCarYawPara;
  Eigen::Vector3d UTMFixPosition;
  Eigen::Quaterniond UTMFixPose;
  nav_msgs::Odometry utmOdom;

  Eigen::Vector3d lasttransformAftMapped;
  Eigen::Vector3d VehicleVelocity;
  double lastTimestamp;

  std_msgs::Header currentHeader;

  string FileDir = "/home/kyle/Downloads/ROSfile/";
  string FileName = FileDir+"Odometry.txt";   
//   ofstream fout;

 public:
  pcl::PointCloud<PointType>::Ptr FusiontoUTMforTest; 
  TransformFusion() : nh("~") {
    pubLaserOdometry2 =
        nh.advertise<nav_msgs::Odometry>("/integrated_to_init", 5);
    subLaserOdometry = nh.subscribe<nav_msgs::Odometry>(
        "/laser_odom_to_init", 5, &TransformFusion::laserOdometryHandler, this);
    subOdomAftMapped = nh.subscribe<nav_msgs::Odometry>(
        "/aft_mapped_to_init", 5, &TransformFusion::odomAftMappedHandler, this);

    pubUTMOdom = nh.advertise<nav_msgs::Odometry>("/utm_odom_to_init", 5);

    pubGPSforCtrl = nh.advertise<sleipnir_msgs::sensorgps>("/localization/fusion_position", 5);
    // pubGPSforCtrl = nh.advertise<sleipnir_msgs::gps4control>("/gps4control", 5);

    // nh.param<double>("/OriLon", OriLon, 113.387985229);
    // nh.param<double>("/OriLat", OriLat, 23.040807724);
    // nh.param<double>("/OriAlt", OriAlt, 2.96000003815);
    // nh.param<double>("/OriYaw", OriYaw, 76.1139984131);
    // nh.param<double>("/OriPitch", OriPitch, -1.33500003815);
    // nh.param<double>("/OriRoll", OriRoll, 1.82000005245);
    // nh.param<double>("/compensate_init_yaw", compensate_init_yaw, 0.0);
    // nh.param<double>("/compensate_init_pitch", compensate_init_pitch, 0.0);
    // nh.param<double>("/compensate_init_roll", compensate_init_roll, 0.0);
    // nh.param<double>("/mappingCarYawPara", mappingCarYawPara, 0.000);
    parameter::readInitPose(nh);

    // cout<<"OriAlt:"<<OriAlt<<endl;

    laserOdometry2.header.frame_id = "camera_init";
    laserOdometry2.child_frame_id = "loam_camera";

    laserOdometryTrans2.frame_id_ = "camera_init";
    laserOdometryTrans2.child_frame_id_ = "loam_camera";

    map_2_camera_init_Trans.frame_id_ = "map";
    map_2_camera_init_Trans.child_frame_id_ = "camera_init";

    camera_2_vehicle_Trans.frame_id_ = "loam_camera";
    camera_2_vehicle_Trans.child_frame_id_ = "vehicle";

    utmOdom.header.frame_id = "camera_init";
    utmOdom.child_frame_id = "loam_camera";

    for (int i = 0; i < 6; ++i) {
      transformSum[i] = 0;
      transformIncre[i] = 0;
      transformMapped[i] = 0;
      transformBefMapped[i] = 0;
      transformAftMapped[i] = 0;
    }

    FusiontoUTMforTest.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization

    // fout.open(FileName);
    // fout.clear();
  }

  ~TransformFusion() {
    //   fout.close();
    //   pcl::io::savePCDFileASCII ("/home/kyle/ros/kyle_ws/src/lins-gps-iris/pcd/FusiontoUTM.pcd", *(FusiontoUTMforTest));
  }

//   该函数更改了transformIncre（平移增量）和transformMapped（经过mapping矫正过后的最终的世界坐标系
//   下的位姿），用来融合odometry节点mapping节点后的位姿
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
    transformMapped[0] = -asin(srx);

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
    transformMapped[1] = atan2(srycrx / cos(transformMapped[0]),
                               crycrx / cos(transformMapped[0]));

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
    transformMapped[2] = atan2(srzcrx / cos(transformMapped[0]),
                               crzcrx / cos(transformMapped[0]));

    x1 = cos(transformMapped[2]) * transformIncre[3] -
         sin(transformMapped[2]) * transformIncre[4];
    y1 = sin(transformMapped[2]) * transformIncre[3] +
         cos(transformMapped[2]) * transformIncre[4];
    z1 = transformIncre[5];

    x2 = x1;
    y2 = cos(transformMapped[0]) * y1 - sin(transformMapped[0]) * z1;
    z2 = sin(transformMapped[0]) * y1 + cos(transformMapped[0]) * z1;

    transformMapped[3] = transformAftMapped[3] - (cos(transformMapped[1]) * x2 +
                                                  sin(transformMapped[1]) * z2);
    transformMapped[4] = transformAftMapped[4] - y2;
    transformMapped[5] =
        transformAftMapped[5] -
        (-sin(transformMapped[1]) * x2 + cos(transformMapped[1]) * z2);
  }

  void TranformToGPS(const double timestamp)
  {
    // get the GPS origination
    // 我们车上使用的星网宇达GPS接收器，他的航向角方向与笛卡尔直角坐标系相反
    // 加上外参，初始姿态转到雷达坐标系下
    Eigen::Vector3d  InitEulerAngle=Eigen::Vector3d((-OriYaw+90.0)*deg+mappingCarYawPara, -OriPitch*deg, OriRoll*deg);
    // cout<<"mappingCarYawPara: "<<mappingCarYawPara<<endl;
    Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(InitEulerAngle(2),Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(InitEulerAngle(1),Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(InitEulerAngle(0),Eigen::Vector3d::UnitZ()));
    // init_fix_odom_pose = yawAngle*pitchAngle*rollAngle;
    Eigen::Quaterniond init_fix_odom_pose;
    init_fix_odom_pose = yawAngle;
    // init_fix_odom_pose = init_fix_odom_pose.inverse();

    // in /map
    double northing, easting;
    std::string zone;
    // LLtoUTM(OriLat, OriLon, northing, easting, zone);
    bool northp;
    int izone;
    UTMUPS::Forward(OriLat, OriLon, izone, northp, easting, northing);

    Eigen::Vector3d fix_odom_position;
    fix_odom_position.x() = easting;
    fix_odom_position.y() = northing;
    fix_odom_position.z() = OriAlt;
    // cout<<setprecision(12)<<"fix_odom_position: ("<<fix_odom_position.transpose()<<")"<<endl;

    // fix_odom_position = init_fix_odom_pose * fix_odom_position;

    // in /camera_init
    UTMFixPosition.x() = fix_odom_position.y();
    UTMFixPosition.y() = fix_odom_position.z();
    UTMFixPosition.z() = fix_odom_position.x();
    // cout<<setprecision(12)<<"UTMFixPosition: ("<<UTMFixPosition.transpose()<<")"<<endl;
    
    UTMFixPose.x() = init_fix_odom_pose.y();
    UTMFixPose.y() = init_fix_odom_pose.z();
    UTMFixPose.z() = init_fix_odom_pose.x();
    UTMFixPose.w() = init_fix_odom_pose.w();

    Eigen::AngleAxisd rv_transformTobeMapped;
    // 减去外参，转到GPS坐标系
    rv_transformTobeMapped=Eigen::AngleAxisd(transformMapped[2],Vector3d::UnitZ())*
                            Eigen::AngleAxisd(transformMapped[0],Vector3d::UnitX())*
                            // 这里应该是减去外参
                            Eigen::AngleAxisd(transformMapped[1]-mappingCarYawPara,Vector3d::UnitY());
    Eigen::Quaterniond Q_transformTobeMapped = Eigen::Quaterniond(rv_transformTobeMapped);
    Eigen::Vector3d V_transformTobeMapped(transformMapped[3], transformMapped[4], transformMapped[5]);

    VehicleVelocity = (V_transformTobeMapped - lasttransformAftMapped)/(timestamp-lastTimestamp);
    lasttransformAftMapped = V_transformTobeMapped;
    lastTimestamp = timestamp;
    // cout<<setprecision(12)<<"V_transformTobeMapped: ("<<V_transformTobeMapped.transpose()<<")"<<endl;

    UTMFixPosition = UTMFixPosition + UTMFixPose*V_transformTobeMapped;
    // 跟前面的外参抵消，回到激光雷达坐标系
    UTMFixPose = UTMFixPose * Q_transformTobeMapped;
    Eigen::Vector3d EularUTMFixPose = UTMFixPose.toRotationMatrix().eulerAngles(1, 0, 2);
    // cout<<setprecision(12)<<"UTMFixPose: ("<<UTMFixPose.toRotationMatrix().eulerAngles(1, 0, 2).transpose()*rad<<")"<<endl;

    double aftLon, aftLat;

    // zone这里是空的，如果使用的话要注意
    // UTMtoLL(UTMFixPosition.x(), UTMFixPosition.z(), zone, aftLat, aftLon);
    
    UTMUPS::Reverse(izone, northp, UTMFixPosition.z(), UTMFixPosition.x(), aftLat, aftLon);
    // cout<<"aftLat: "<<aftLat<<" "<<"aftLon: "<<aftLon<<endl;

    sleipnir_msgs::sensorgps gps4control_msg;
    // sleipnir_msgs::gps4control gps4control_msg;
    // gps4control_msg.timestamp = timestamp;
    gps4control_msg.header.stamp = ros::Time().fromSec(timestamp);

    gps4control_msg.lat = aftLat;
    gps4control_msg.lon = aftLon;
    gps4control_msg.status = 4;

    // gps4control_msg.isvalid = 1;
    gps4control_msg.satenum = 50;
    gps4control_msg.x = UTMFixPosition.z();
    gps4control_msg.y = UTMFixPosition.x();

    gps4control_msg.heading = EularUTMFixPose[0]*rad;
    if(abs(EularUTMFixPose[2])*rad>90 || abs(EularUTMFixPose[1])*rad>90)
    {
        gps4control_msg.heading = EularUTMFixPose[0]*rad - 180;
    }
    gps4control_msg.heading = 360-(gps4control_msg.heading+180)-90;
    if(gps4control_msg.heading<0) gps4control_msg.heading += 360;

    gps4control_msg.velocity = VehicleVelocity.norm();
    // gps4control_msg.heading = 
    pubGPSforCtrl.publish(gps4control_msg);

    // PointType FusionPoint;
    // FusionPoint.x = UTMFixPosition.z()- 700000.00;;
    // FusionPoint.y = UTMFixPosition.x()- 2400000.00;
    // FusiontoUTMforTest->push_back(FusionPoint);

    // cout<<"transformMapped : "<<transformMapped[0]<<","<<transformMapped[1]<<","<<transformMapped[2]<<endl;
    // cout<<setprecision(12)<<"UTMFixPosition: ("<<aftLat<<","<<aftLon<<")"<<endl<<endl;

  }

  void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr& laserOdometry) {
    currentHeader = laserOdometry->header;

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

    // /odom -> /map，得到transformMapped
    // 该函数更改了transformIncre（平移增量）和transformMapped（经过mapping矫正过后的最终
    // 的世界坐标系下的位姿），用来融合odometry节点mapping节点后的位姿
    transformAssociateToMap();

    TranformToGPS(laserOdometry->header.stamp.toSec());

    // transform the transformMapped[] to WGS84 and then add it to GPS origination

    Eigen::Matrix<double, 6, 1> Vec_transformMapped;

    Vec_transformMapped[0] = transformMapped[5];
    Vec_transformMapped[1] = transformMapped[3];
    Vec_transformMapped[2] = transformMapped[4];
    Vec_transformMapped[3] = transformMapped[2] * rad;
    Vec_transformMapped[4] = transformMapped[0] * rad;
    Vec_transformMapped[5] = transformMapped[1] * rad;

    // Eigen::Affine3d T_matrix = Eigen::Affine3d::Identity();
    // Eigen::AngleAxisd AAroll = Eigen::AngleAxisd(transformMapped[2], Vector3d::UnitX());
    // Eigen::AngleAxisd AApitch = Eigen::AngleAxisd(transformMapped[0], Vector3d::UnitY());
    // Eigen::AngleAxisd AAyaw = Eigen::AngleAxisd(transformMapped[1], Vector3d::UnitZ());
    // Eigen::Quaterniond E_geoQuat;
    // E_geoQuat = AAroll * AApitch * AAyaw;
    // Eigen::Vector3d E_transition(transformMapped[5], transformMapped[3], transformMapped[4]);
    // T_matrix.rotate(E_geoQuat);
    // T_matrix.pretranslate(E_transition);

    // fout<<laserOdometry->header.stamp.toNSec()<<endl;
    // // fout<<Vec_transformMapped.transpose()<<endl;
    // fout<<T_matrix.matrix()<<endl;

    geoQuat = tf::createQuaternionMsgFromRollPitchYaw(
        transformMapped[2], -transformMapped[0], -transformMapped[1]);

    laserOdometry2.header.stamp = laserOdometry->header.stamp;
    laserOdometry2.pose.pose.orientation.x = -geoQuat.y;
    laserOdometry2.pose.pose.orientation.y = -geoQuat.z;
    laserOdometry2.pose.pose.orientation.z = geoQuat.x;
    laserOdometry2.pose.pose.orientation.w = geoQuat.w;
    laserOdometry2.pose.pose.position.x = transformMapped[3];
    laserOdometry2.pose.pose.position.y = transformMapped[4];
    laserOdometry2.pose.pose.position.z = transformMapped[5];
    pubLaserOdometry2.publish(laserOdometry2);

    laserOdometryTrans2.stamp_ = laserOdometry->header.stamp;
    laserOdometryTrans2.setRotation(
        tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
    laserOdometryTrans2.setOrigin(tf::Vector3(
        transformMapped[3], transformMapped[4], transformMapped[5]));
    tfBroadcaster2.sendTransform(laserOdometryTrans2);
  }

  void odomAftMappedHandler(const nav_msgs::Odometry::ConstPtr& odomAftMapped) {
    double roll, pitch, yaw;
    geometry_msgs::Quaternion geoQuat = odomAftMapped->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w))
        .getRPY(roll, pitch, yaw);

    transformAftMapped[0] = -pitch;
    transformAftMapped[1] = -yaw;
    transformAftMapped[2] = roll;

    transformAftMapped[3] = odomAftMapped->pose.pose.position.x;
    transformAftMapped[4] = odomAftMapped->pose.pose.position.y;
    transformAftMapped[5] = odomAftMapped->pose.pose.position.z;

    transformBefMapped[0] = odomAftMapped->twist.twist.angular.x;
    transformBefMapped[1] = odomAftMapped->twist.twist.angular.y;
    transformBefMapped[2] = odomAftMapped->twist.twist.angular.z;

    transformBefMapped[3] = odomAftMapped->twist.twist.linear.x;
    transformBefMapped[4] = odomAftMapped->twist.twist.linear.y;
    transformBefMapped[5] = odomAftMapped->twist.twist.linear.z;

  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "lego_loam");

  TransformFusion TFusion;

  ROS_INFO("\033[1;32m---->\033[0m Transform Fusion Started.");

  ros::spin();

  return 0;
}
