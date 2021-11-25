#include "utility.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

// #include "sensorgps_msgs/sensorgps.h"
#include <sleipnir_msgs/sensorgps.h>

#include <sensor_msgs/NavSatStatus.h>
#include <sensor_msgs/NavSatFix.h>
#include <gps_common/conversions.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <queue> // For queue container
#include <mutex>

using namespace gtsam;
using namespace std;
using namespace gps_common;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

std::mutex mtx;

ros::Subscriber subImu;
ros::Subscriber subGps;
ros::Subscriber subFixVel;
ros::Subscriber subLaserOdomVel;
ros::Publisher pubImuOdometry;
ros::Publisher pubImuPath;
ros::Publisher fix_position_pub;
ros::Publisher fusion_position_pub;
ros::Publisher pubGPSforCtrl;

bool init_fix = false;
double init_fix_odom_x, init_fix_odom_y, init_fix_odom_z; 
double init_fix_odom_yaw, init_fix_odom_pitch, init_fix_odom_roll;
Eigen::Quaterniond init_fix_odom_pose;
std::string init_fix_zoom;
pcl::PointCloud<PointType>::Ptr GPSHistoryPosition3D;

gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

// std::deque<sensor_msgs::Imu> imuQueOpt;
std::deque<sensor_msgs::Imu> imuQueImu;
// std::deque<sensor_msgs::Imu> imuQueImu2;
std::deque<geometry_msgs::TwistStamped> FixVelQue;

// std::deque<geometry_msgs::TwistStamped> LaserVelQue;

// gtsam::Pose3 prevPose_;
// gtsam::Vector3 prevVel_;
// gtsam::NavState prevState_;
// gtsam::imuBias::ConstantBias prevBias_;

// gtsam::NavState prevStateOdom;
// gtsam::imuBias::ConstantBias prevBiasOdom;

gtsam::ISAM2 optimizer;
gtsam::NonlinearFactorGraph graphFactors;
gtsam::Values graphValues;

// IMU
float imuAccNoise = 3.9939570888238808e-03;
float imuGyrNoise = 1.5636343949698187e-03;
float imuAccBiasN = 6.4356659353532566e-05;
float imuGyrBiasN = 3.5640318696367613e-05;
float imuGravity = 9.80511;

pcl::PointCloud<PointType>::Ptr globalImuOdomPoses;

double curFixTime = 0;
double curImuTime = 0;
double lastImuTime = 0;
Eigen::Vector3d last_fix_odom_position;

double curLaserTime = 0;
double lastLaserTime = 0;

// NonlinearFactorGraph *graph = new NonlinearFactorGraph();

gtsam::Rot3 prevRot_opt;
gtsam::Point3 prevPoint_opt;
gtsam::Pose3 prevPose_opt;
gtsam::Vector3 prevVel_opt;
gtsam::NavState prevState_opt;
gtsam::imuBias::ConstantBias prevBias_opt;

int correction_count = 0;
bool sencondGps = false;
bool gpsValid = true;
int gpsValidNum = 0;
int gpsValidNumThes = 6;
int gpsMeasSkip = 5;
double tranErrThes = 0.7;
double angleErrThes = 5.0;
int key = 0;
gtsam::Point3 lastImuPos(0.0 ,0.0 ,0.0);
Eigen::Vector3d lastGpsPos(0.0 ,0.0 ,0.0 );
Eigen::Vector3d lastGpsVel(0.0 ,0.0 ,0.0);
double planePitchThes = 0.0;
int outlierNum = 0;
// double last_gpsRotPitch;
// Assemble prior noise model and add it the graph.
// 初始点的噪声配置
noiseModel::Diagonal::shared_ptr pose_noise_model = noiseModel::Diagonal::Sigmas((Vector(6) << 0.01, 0.01, 0.01, 0.5, 0.5, 0.5).finished()); // rad,rad,rad,m, m, m
noiseModel::Diagonal::shared_ptr velocity_noise_model = noiseModel::Isotropic::Sigma(3,0.1); // m/s
noiseModel::Diagonal::shared_ptr bias_noise_model = noiseModel::Isotropic::Sigma(6,1e-3);

// 当前点的噪声
noiseModel::Diagonal::shared_ptr curr_pose_noise_model = noiseModel::Diagonal::Sigmas((Vector(6) << 0.01, 0.01, 0.01, 0.5, 0.5, 0.5).finished()); // rad,rad,rad,m, m, m
noiseModel::Diagonal::shared_ptr curr_velocity_noise_model = noiseModel::Isotropic::Sigma(3,0.1); // m/s
noiseModel::Diagonal::shared_ptr curr_bias_noise_model = noiseModel::Isotropic::Sigma(6,1e-3);

gtsam::noiseModel::Diagonal::shared_ptr correctionNoise; // meter
pcl::PointCloud<PointType>::Ptr globalFusionOdomPoses;

ofstream fout_evo;
ofstream fout_gps_evo;

Eigen::Vector3d UTMFixPosition;
Eigen::Quaterniond UTMFixPose;

void fixHandler(const sensor_msgs::NavSatFixConstPtr& msg)
{
    // ROS_DEBUG("have received a fix msg!");
    // std::lock_guard<std::mutex> lock(mtx);
    // save the gps data
    sensor_msgs::NavSatFix thisFix = *msg;
    // current gps data timestamp
    curFixTime = thisFix.header.stamp.toSec();

    sensor_msgs::Imu InitImu;
    if(imuQueImu.empty()) return;

    // --------------------------------------------------------
    // if(imuQueImu.empty()) return;
    // else
    // {
    //     InitImu = imuQueImu.back();
    //     init_fix_odom_pose.w() = InitImu.orientation.w;
    //     init_fix_odom_pose.x() = InitImu.orientation.x;
    //     init_fix_odom_pose.y() = InitImu.orientation.y;
    //     init_fix_odom_pose.z() = InitImu.orientation.z;
    //     init_fix_odom_yaw = init_fix_odom_pose.toRotationMatrix().eulerAngles(0, 1, 2)[2];
    // }
    // sleipnir_msgs::sensorgps gps4control_msg;
    // gps4control_msg.header.stamp = thisFix.header.stamp;
    // gps4control_msg.lat = thisFix.latitude;
    // gps4control_msg.lon = thisFix.longitude;
    // gps4control_msg.status = '3';
    // gps4control_msg.satenum = 50;
    // // gps4control_msg.x = init_fix_odom_x;
    // // gps4control_msg.y = init_fix_odom_y;
    // gps4control_msg.heading = init_fix_odom_yaw*DEGREES_PER_RADIAN;
    // if(abs(init_fix_odom_pose.toRotationMatrix().eulerAngles(0, 1, 2)[0])*DEGREES_PER_RADIAN>90 || abs(init_fix_odom_pose.toRotationMatrix().eulerAngles(0, 1, 2)[1])*DEGREES_PER_RADIAN>90)
    // {
    //     gps4control_msg.heading = init_fix_odom_yaw*DEGREES_PER_RADIAN - 180;
    // }
    // gps4control_msg.heading = 360-(gps4control_msg.heading+180)-90;
    // if(gps4control_msg.heading<0) gps4control_msg.heading+360;
    // // gps4control_msg.velocity = VehicleVelocity.norm();
    // pubGPSforCtrl.publish(gps4control_msg);
    // return;
    // ---------------------------------------------------------

    // Initial the pose and position
    if(!init_fix)
    {
        // find the beginning imu
        double InitImuTime = -1.0;
        // 截取IMU队列
        while(!imuQueImu.empty())
        {
            InitImuTime = imuQueImu.front().header.stamp.toSec();
            if(InitImuTime < curFixTime - 0.02)
            {
                imuQueImu.pop_front();
            }
            else
                break;
        }
        // can not find the proper imu data to be the initial pose, exit and wait for the next gps
        if(imuQueImu.empty()) return;
        // get the corresponding imu data and set the initial pose using it
        InitImu = imuQueImu.front();
        imuQueImu.pop_front();
        lastImuTime = InitImu.header.stamp.toSec();

        // set begining pose and position
        // 从IMU获取姿态作为GPS测量的姿态
        init_fix_odom_pose.w() = InitImu.orientation.w;
        init_fix_odom_pose.x() = InitImu.orientation.x;
        init_fix_odom_pose.y() = InitImu.orientation.y;
        init_fix_odom_pose.z() = InitImu.orientation.z;
        // cout<<"init_fix_odom_pose"<<init_fix_odom_pose.toRotationMatrix().eulerAngles(2,1,0).transpose()*180/3.1415926<<endl;
        // project the position from wgs84 to utm
        // 将GPS测量转到UTM
        LLtoUTM(thisFix.latitude, thisFix.longitude, init_fix_odom_y, init_fix_odom_x, init_fix_zoom);
        init_fix_odom_z = thisFix.altitude;
        init_fix_odom_pose = init_fix_odom_pose.inverse();

        init_fix_odom_yaw = init_fix_odom_pose.toRotationMatrix().eulerAngles(0, 1, 2)[2];

        if(FixVelQue.empty()) return;
        geometry_msgs::TwistStamped InitVel = FixVelQue.back();
        FixVelQue.pop_back();
        Eigen::Vector3d EigenInitVel(InitVel.twist.linear.x,
                                    InitVel.twist.linear.y,
                                    InitVel.twist.linear.z);
        // EigenInitVel = init_fix_odom_pose * EigenInitVel;

        // Store previous state for the imu integration and the latest predicted outcome.
        // 将第一帧GPS的状态存到GTSAM中
        prevRot_opt = gtsam::Rot3::Quaternion(1.0, 0.0, 0.0, 0.0); // 初始姿态
        prevPoint_opt = gtsam::Point3(0.0, 0.0, 0.0); // 初始位置
        prevPose_opt = gtsam::Pose3(prevRot_opt, prevPoint_opt); // 初始位姿
        prevVel_opt = gtsam::Vector3(EigenInitVel); // 初始速度
        // for preintegration
        prevState_opt = gtsam::NavState(prevPose_opt, prevVel_opt); // 初始状态

        // 设置初值
        graphValues.insert(X(correction_count), prevPose_opt);
        graphValues.insert(V(correction_count), prevVel_opt);
        graphValues.insert(B(correction_count), prevBias_opt);

        // Add all prior factors (pose, velocity, bias) to the graph.
        // NonlinearFactorGraph *graph = new NonlinearFactorGraph();
        // 添加一元因子
        graphFactors.add(PriorFactor<Pose3>(X(correction_count), prevPose_opt, pose_noise_model));
        graphFactors.add(PriorFactor<Vector3>(V(correction_count), prevVel_opt,velocity_noise_model));
        graphFactors.add(PriorFactor<imuBias::ConstantBias>(B(correction_count), prevBias_opt,bias_noise_model));

        // optimize once
        optimizer.update(graphFactors, graphValues);
        graphFactors.resize(0);
        graphValues.clear();

        imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_opt);

        init_fix = true;

        sleipnir_msgs::sensorgps gps4control_msg;
        gps4control_msg.header.stamp = thisFix.header.stamp;
        gps4control_msg.lat = thisFix.latitude;
        gps4control_msg.lon = thisFix.longitude;
        gps4control_msg.status = '3';
        gps4control_msg.satenum = 50;
        gps4control_msg.x = init_fix_odom_x;
        gps4control_msg.y = init_fix_odom_y;
        gps4control_msg.heading = -init_fix_odom_yaw*DEGREES_PER_RADIAN;
        if(abs(init_fix_odom_pose.toRotationMatrix().eulerAngles(0, 1, 2)[0])*DEGREES_PER_RADIAN>90 || abs(init_fix_odom_pose.toRotationMatrix().eulerAngles(0, 1, 2)[1])*DEGREES_PER_RADIAN>90)
        {
            gps4control_msg.heading = -init_fix_odom_yaw*DEGREES_PER_RADIAN - 180;
        }
        gps4control_msg.heading = 360-(gps4control_msg.heading+180)-90;
        if(gps4control_msg.heading<0) gps4control_msg.heading+360;
        // gps4control_msg.velocity = VehicleVelocity.norm();
        pubGPSforCtrl.publish(gps4control_msg);
    }// end initial
    
    //for testing-------------------------------------------------------------------------
    // transform the gps data from wgs84 to utm
    double northing, easting;
    std::string zone;
    LLtoUTM(thisFix.latitude, thisFix.longitude, northing, easting, zone);
    nav_msgs::Odometry fix_odom;
    fix_odom.header.stamp = thisFix.header.stamp;
    Eigen::Vector3d fix_odom_position;
    fix_odom_position.x() = easting - init_fix_odom_x;
    fix_odom_position.y() = northing - init_fix_odom_y;
    fix_odom_position.z() = thisFix.altitude - init_fix_odom_z;
    fix_odom_position = Eigen::AngleAxisd(init_fix_odom_yaw+3.1415926,Eigen::Vector3d::UnitZ()) * fix_odom_position;

    // traject of gps
    PointType gps_position;
    gps_position.x = fix_odom_position.x();
    gps_position.y = fix_odom_position.y();
    gps_position.z = fix_odom_position.z();

    // ROS_DEBUG("have published a gps position!\n");

    sensor_msgs::Imu FixImu;
    //for testing-----------------------------------------------------------------------

    // sensor fusion
    if(sencondGps)
    {
        // performing preintegration
        sensor_msgs::Imu thisImu;
        while(!imuQueImu.empty())
        {
            if(imuQueImu.front().header.stamp.toSec()<thisFix.header.stamp.toSec()-0.02)
            {
                // extract the early imu
                thisImu = imuQueImu.front();
                imuQueImu.pop_front();
                curImuTime = thisImu.header.stamp.toSec();
                double Imu_dt = curImuTime - lastImuTime;
                // cout<<"dt:"<<Imu_dt<<endl;
                lastImuTime = curImuTime;
                // integrate this single imu message
                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                        gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), Imu_dt);
            }
            else
                break;
        }

        // 从IMU获取姿态作为GPS测量的姿态
        // 当前GPS对应姿态
        Eigen::Quaterniond gpsRot(thisImu.orientation.w,
                                thisImu.orientation.x,
                                thisImu.orientation.y,
                                thisImu.orientation.z);
        // 转到笛卡尔直角坐标系（为什么这里还要把IMU预测得到的位姿作为一元因子融合？）
        gpsRot = init_fix_odom_pose * gpsRot;
        gtsam::Rot3 curGpsRot = gtsam::Rot3::Quaternion(gpsRot.w(),
                                                        gpsRot.x(),
                                                        gpsRot.y(),
                                                        gpsRot.z());

        // find the corresponding /fix/vel
        // 获取当前GPS测量对应的速度数据
        // ROS_DEBUG("before FixVelQue: %d\n", FixVelQue.size());
        double CurrFixVelTime = -1.0;

        ROS_DEBUG("after FixVelQue: %d\n", FixVelQue.size());
        if(FixVelQue.empty()) return;
        geometry_msgs::TwistStamped CurrVel = FixVelQue.back();
        FixVelQue.pop_back();
        Eigen::Vector3d EigenCurrVel(CurrVel.twist.linear.x,
                                    CurrVel.twist.linear.y,
                                    CurrVel.twist.linear.z);
        EigenCurrVel = curGpsRot * EigenCurrVel;

        // generate imu factor
        // 生成imu预积分因子
        const PreintegratedImuMeasurements& preint_imu =
            dynamic_cast<const PreintegratedImuMeasurements&>(
              *imuIntegratorImu_);
        ImuFactor imu_factor(X(correction_count-1), V(correction_count-1),
                             X(correction_count  ), V(correction_count  ),
                             B(correction_count-1),
                             preint_imu);
        graphFactors.add(imu_factor);
        imuBias::ConstantBias zero_bias(Vector3(0, 0, 0), Vector3(0, 0, 0));
        graphFactors.add(BetweenFactor<imuBias::ConstantBias>(B(correction_count-1), B(correction_count),
                                            gtsam::imuBias::ConstantBias(), bias_noise_model));

        // imu predict
        gtsam::NavState propState_ = imuIntegratorImu_->predict(prevState_opt, prevBias_opt);

        // using imu preintegration to predict the transformation between two key frame
        // compute the transformation between two gps measurement
        // --------------------------------------------------------------------------------
        // judge the gps outlier using amplitude and prase
        // 用于判断GPS的异常
        Eigen::Vector3d ImuTran(propState_.position()-lastImuPos);
        Eigen::Vector3d FixTran(fix_odom_position-lastGpsPos);

        double magErr = abs(ImuTran.norm() / FixTran.norm()-1);
        // cout<<"magErr: "<<magErr<<endl;
        double angleErr = acos(ImuTran.dot(FixTran)/(ImuTran.norm()*FixTran.norm()))*180/PI;
        // cout<<"angleErr: "<<angleErr<<endl<<endl;
        // -----------------------------------------------------------------------------------

        Eigen::Vector3d Gravity(0.0, 0.0, -9.81);
        // cout<<"EigenCurrVel: "<<EigenCurrVel.transpose()<<endl;
        Eigen::Vector3d gps_DeltaP_ij = curGpsRot.inverse()*(fix_odom_position-lastGpsPos-(0.1*EigenCurrVel)-0.5*Gravity*0.1*0.1);
        // cout<<"gps_DeltaP_ij: "<<gps_DeltaP_ij.transpose()<<endl;
        // cout<<"imu_DeltaP_ij"<<imuIntegratorImu_->deltaPij().transpose()<<endl;
        cout<<"diff DeltaP_ij: "<<(gps_DeltaP_ij-imuIntegratorImu_->deltaPij()).norm()<<endl;
        double chi_score = (gps_DeltaP_ij-imuIntegratorImu_->deltaPij()).norm();

        lastImuPos = propState_.position();
        lastGpsPos = fix_odom_position;

        // 使用鲁棒核函数约束outliers
        if(chi_score<0.2)
        {
            gpsValid = true;
            correctionNoise = noiseModel::Diagonal::Sigmas((Vector(6) <<0.1, 0.1, 0.1, 1.0, 1.0, 300.0).finished()); // rad,rad,rad,m, m, m
            if(outlierNum>0)
            {
                outlierNum--;
                goto outliers_handler;
            }
        }
        else
        {
            outlierNum = 30;
        outliers_handler:
            gpsValid = false;
            Eigen::Vector3d vehicle_vel(propState_.velocity().norm(), 0.0, 0.0);
            noiseModel::Diagonal::shared_ptr temp_velocity_noise_model = noiseModel::Isotropic::Sigma(3,1.0);
            gtsam::PriorFactor<gtsam::Vector3> vehicle_vel_factor(V(correction_count),
                                                                    gtsam::Vector3(curGpsRot*vehicle_vel), temp_velocity_noise_model);
            correctionNoise = noiseModel::Diagonal::Sigmas((Vector(6) <<0.1, 0.1, 0.1, 50.0, 50.0, 300.0).finished()); // rad,rad,rad,m, m, m
        }
        auto huber = gtsam::noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(0.01), correctionNoise);

        gtsam::Point3 curFixPoint = gtsam::Vector3(fix_odom_position);
        gtsam::Rot3 curFixRot = curGpsRot;
        gtsam::Pose3 curFixPose(curFixRot, curFixPoint);
        // gtsam::GPSFactor gps_factor(X(correction_count), curFixPoint, huber);
        gtsam::PriorFactor<gtsam::Pose3> gps_factor(X(correction_count), curFixPose, huber);
        graphFactors.add(gps_factor);

        // insert predicted values
        graphValues.insert(X(correction_count), propState_.pose());
        graphValues.insert(V(correction_count), propState_.v());
        graphValues.insert(B(correction_count), prevBias_opt);

        // optimize
        optimizer.update(graphFactors, graphValues);
        optimizer.update();
        optimizer.update();
        optimizer.update();
        optimizer.update();
        graphFactors.resize(0);
        graphValues.clear();

        // Overwrite the beginning of the preintegration for the next step.
        gtsam::Values result = optimizer.calculateEstimate();
        prevState_opt = NavState(result.at<Pose3>(X(correction_count)),
                                result.at<Vector3>(V(correction_count)));
        prevBias_opt = result.at<imuBias::ConstantBias>(B(correction_count));

        gtsam::Pose3 fusionPose = gtsam::Pose3(prevState_opt.quaternion(), prevState_opt.position());


        // 这里发布sensorgps
        UTMFixPosition.x() = init_fix_odom_x;
        UTMFixPosition.y() = init_fix_odom_y;
        UTMFixPosition.z() = init_fix_odom_z;
        UTMFixPose = init_fix_odom_pose.inverse()*gpsRot;
        UTMFixPosition = UTMFixPosition + init_fix_odom_pose.inverse()*Eigen::Vector3d(fusionPose.translation().x(), 
                                                                        fusionPose.translation().y(),
                                                                        fusionPose.translation().z());
        Eigen::Vector3d EularUTMFixPose = UTMFixPose.toRotationMatrix().eulerAngles(0,1,2);
        double aftLon, aftLat;
        UTMtoLL(UTMFixPosition.y(), UTMFixPosition.x(), zone, aftLat, aftLon);

        sleipnir_msgs::sensorgps gps4control_msg;
        gps4control_msg.header.stamp = thisFix.header.stamp;
        gps4control_msg.lat = aftLat;
        gps4control_msg.lon = aftLon;
        gps4control_msg.status = '3';
        gps4control_msg.satenum = 50;
        gps4control_msg.x = UTMFixPosition.x();
        gps4control_msg.y = UTMFixPosition.y();

        gps4control_msg.heading = EularUTMFixPose[2]*DEGREES_PER_RADIAN;
        if(abs(EularUTMFixPose[0])*DEGREES_PER_RADIAN>90 || abs(EularUTMFixPose[1])*DEGREES_PER_RADIAN>90)
        {
            gps4control_msg.heading = EularUTMFixPose[2]*DEGREES_PER_RADIAN - 180;
        }
        gps4control_msg.heading = 360-(gps4control_msg.heading+180)-90;
        if(gps4control_msg.heading<0) gps4control_msg.heading+360;
        // gps4control_msg.velocity = VehicleVelocity.norm();
        pubGPSforCtrl.publish(gps4control_msg);
        

        // Reset the preintegration object.
        imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_opt);

        // pub the fusion localization trajectoty
        PointType statePoint;
        statePoint.x = fusionPose.translation().x();
        statePoint.y = fusionPose.translation().y();
        statePoint.z = fusionPose.translation().z();
        if(gpsValid) 
            statePoint.intensity = 255;
        else
            statePoint.intensity = 0;
        globalFusionOdomPoses->push_back(statePoint);
        sensor_msgs::PointCloud2 fusion_position_cloudMsgTemp;
        pcl::toROSMsg(*globalFusionOdomPoses, fusion_position_cloudMsgTemp);
        fusion_position_cloudMsgTemp.header.stamp = thisFix.header.stamp;
        fusion_position_cloudMsgTemp.header.frame_id = "local_map";
        fusion_position_pub.publish(fusion_position_cloudMsgTemp);

        gps_position.intensity = statePoint.intensity;
        GPSHistoryPosition3D->push_back(gps_position);
        sensor_msgs::PointCloud2 gps_position_cloudMsgTemp;
        pcl::toROSMsg(*GPSHistoryPosition3D, gps_position_cloudMsgTemp);
        gps_position_cloudMsgTemp.header.stamp = thisFix.header.stamp;
        gps_position_cloudMsgTemp.header.frame_id = "local_map";
        fix_position_pub.publish(gps_position_cloudMsgTemp);
        
        // 实例化一个tf广播器
        static tf::TransformBroadcaster tfFusionBroadcaster;
        geometry_msgs::TransformStamped aftFusionTrans;
        aftFusionTrans.header.stamp = thisFix.header.stamp;
        aftFusionTrans.header.frame_id = "local_map";
        aftFusionTrans.child_frame_id = "odom_fusion";
        aftFusionTrans.transform.rotation.w =  fusionPose.rotation().toQuaternion().w();
        aftFusionTrans.transform.rotation.x =  fusionPose.rotation().toQuaternion().x();
        aftFusionTrans.transform.rotation.y =  fusionPose.rotation().toQuaternion().y();
        aftFusionTrans.transform.rotation.z =  fusionPose.rotation().toQuaternion().z();
        aftFusionTrans.transform.translation.x = fusionPose.translation().x();
        aftFusionTrans.transform.translation.y = fusionPose.translation().y();
        aftFusionTrans.transform.translation.z = fusionPose.translation().z();
        tfFusionBroadcaster.sendTransform(aftFusionTrans);

        fout_evo<<thisFix.header.stamp.toSec()<<" "
              <<fusionPose.translation().x()<<" "
              <<fusionPose.translation().y()<<" "
              <<fusionPose.translation().z()<<" "
              <<fusionPose.rotation().toQuaternion().x()<<" "
              <<fusionPose.rotation().toQuaternion().y()<<" "
              <<fusionPose.rotation().toQuaternion().z()<<" "
              <<fusionPose.rotation().toQuaternion().w()<<endl;

        FixImu = thisImu;
    }
    else
    {
        FixImu = InitImu;
        last_fix_odom_position = fix_odom_position;
        sencondGps = true;
    }
    // 实例化一个tf广播器
    Eigen::Quaterniond FixRot(FixImu.orientation.w, FixImu.orientation.x, FixImu.orientation.y, FixImu.orientation.z);
    FixRot = init_fix_odom_pose * FixRot;
    static tf::TransformBroadcaster tfBroadcaster;
    geometry_msgs::TransformStamped aftFixTrans;
    aftFixTrans.header.stamp = thisFix.header.stamp;
    aftFixTrans.header.frame_id = "local_map";
    aftFixTrans.child_frame_id = "odom_fix";
    aftFixTrans.transform.rotation.w =  FixRot.w();
    aftFixTrans.transform.rotation.x =  FixRot.x();
    aftFixTrans.transform.rotation.y =  FixRot.y();
    aftFixTrans.transform.rotation.z =  FixRot.z();
    aftFixTrans.transform.translation.x = fix_odom_position.x();
    aftFixTrans.transform.translation.y = fix_odom_position.y();
    aftFixTrans.transform.translation.z = fix_odom_position.z();
    tfBroadcaster.sendTransform(aftFixTrans);
    correction_count++;
    key++;

    fout_gps_evo<<thisFix.header.stamp.toSec()<<" "
                      <<fix_odom_position.x()<<" "
                      <<fix_odom_position.y()<<" "
                      <<fix_odom_position.z()<<" "
                      <<FixRot.x()<<" "
                      <<FixRot.y()<<" "
                      <<FixRot.z()<<" "
                      <<FixRot.w()<<endl;
}

void imuHandler(const sensor_msgs::Imu::ConstPtr& imu_raw)
{
    // ROS_DEBUG("imu handler\n");
    std::lock_guard<std::mutex> lock(mtx);

    sensor_msgs::Imu thisImu = *imu_raw;

    // imuQueOpt.push_back(thisImu);
    imuQueImu.push_back(thisImu);
    // imuQueImu2.push_back(thisImu);
}

bool InitVelFlag = false;
Eigen::Quaterniond init_fix_vel_pose;
double curVelImuTime = 0;
double lastVelImuTime = 0;
double velImu_dt = 0;
void FixVelHandle(const geometry_msgs::TwistStamped::ConstPtr& msg)
{
    std::lock_guard<std::mutex> lock(mtx);
    // save the latest data
    geometry_msgs::TwistStamped thisFixVel = *msg;
    FixVelQue.push_back(thisFixVel);
}

// void LaserVelHandle(const geometry_msgs::TwistStamped::ConstPtr& msg)
// {
//     std::lock_guard<std::mutex> lock(mtx);
//     LaserVelQue.push_back(*msg);
// }

void resetOptimization()
{
    gtsam::ISAM2Params optParameters;
    optParameters.relinearizeThreshold = 0.1;
    optParameters.relinearizeSkip = 1;
    optimizer = gtsam::ISAM2(optParameters);

    gtsam::NonlinearFactorGraph newGraphFactors;
    graphFactors = newGraphFactors;

    gtsam::Values NewGraphValues;
    graphValues = NewGraphValues;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "roboat_loam");
    ros::NodeHandle nh;
    ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");

    // 配置预积分器
    boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
    p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous
    p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
    p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities
    gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias
    imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
    imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_opt);

    // 用于可视化
    globalImuOdomPoses.reset(new pcl::PointCloud<PointType>());
    GPSHistoryPosition3D.reset(new pcl::PointCloud<PointType>());
    globalFusionOdomPoses.reset(new pcl::PointCloud<PointType>());

    resetOptimization();

    nh.param<int>("gpsValidNumThes", gpsValidNumThes, 6);
    nh.param<double>("tranErrThes", tranErrThes, 0.7);
    nh.param<double>("angleErrThes", angleErrThes, 5.0);
    nh.param<int>("gpsMeasSkip", gpsMeasSkip, 5);
    nh.param<double>("planePitchThes", planePitchThes, 5.0);

    cout<<"gpsValidNumThes: "<<gpsValidNumThes<<endl;
    cout<<"tranErrThes: "<<tranErrThes<<endl;
    cout<<"angleErrThes: "<<angleErrThes<<endl;
    
    subImu = nh.subscribe<sensor_msgs::Imu>("/imu_correct", 2000, imuHandler);
    subGps = nh.subscribe("/gps/fix", 1000, fixHandler);
    subFixVel = nh.subscribe("/gps/vel", 1000, FixVelHandle);
    // subLaserOdomVel = nh.subscribe("/laser_velocity", 1000, LaserVelHandle);
    pubImuOdometry   = nh.advertise<sensor_msgs::PointCloud2>("/imu/odometry", 100);
    pubImuPath       = nh.advertise<nav_msgs::Path>("/imu/path", 1);
    fix_position_pub = nh.advertise<sensor_msgs::PointCloud2>("/imuodometry/gps_history_position", 2);
    fusion_position_pub = nh.advertise<sensor_msgs::PointCloud2>("/imuodometry/fusion_position", 2);
    pubGPSforCtrl = nh.advertise<sleipnir_msgs::sensorgps>("/sensorgps", 5);

    fout_evo.open("/home/kyle/Downloads/ROSfile/imuOdom/fusion_traj_evo.txt");
    fout_evo.precision(15);
    fout_evo.clear();

    fout_gps_evo.open("/home/kyle/Downloads/ROSfile/imuOdom/gps_traj_evo.txt");
    fout_gps_evo.precision(15);
    fout_gps_evo.clear();

    ros::spin();

    return 0;
}