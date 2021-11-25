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

#include <parameters.h>

using namespace parameter;

class ImageProjection {
 private:
  ros::NodeHandle nh;
  ros::NodeHandle pnh;

  ros::Subscriber subLaserCloud;

  ros::Publisher pubFullCloud;
  ros::Publisher pubFullInfoCloud;

  ros::Publisher pubGroundCloud;
  ros::Publisher pubSegmentedCloud;
  ros::Publisher pubSegmentedCloudPure;
  ros::Publisher pubSegmentedCloudInfo;
  ros::Publisher pubOutlierCloud;

  pcl::PointCloud<PointType>::Ptr laserCloudIn;

  pcl::PointCloud<PointType>::Ptr fullCloud;
  pcl::PointCloud<PointType>::Ptr fullInfoCloud;

  pcl::PointCloud<PointType>::Ptr groundCloud;
  pcl::PointCloud<PointType>::Ptr segmentedCloud;
  pcl::PointCloud<PointType>::Ptr segmentedCloudPure;
  pcl::PointCloud<PointType>::Ptr outlierCloud;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr laserCloudFullResColor;

  PointType nanPoint;

  cv::Mat rangeMat;
  cv::Mat labelMat;
  cv::Mat groundMat;
  int labelCount;

  float startOrientation;
  float endOrientation;

  cloud_msgs::cloud_info segMsg;
  // 存储msg的时间戳和连接的framestd_msgs::Header 
  std_msgs::Header cloudHeader;

  std::vector<std::pair<uint8_t, uint8_t> > neighborIterator;

  uint16_t* allPushedIndX;
  uint16_t* allPushedIndY;

  uint16_t* queueIndX;
  uint16_t* queueIndY;

 public:
  ImageProjection(ros::NodeHandle& nh, ros::NodeHandle& pnh)
      : nh(nh), pnh(pnh) {
    // 订阅激光雷达驱动节点
    subLaserCloud = pnh.subscribe<sensor_msgs::PointCloud2>(
        LIDAR_TOPIC, 1, &ImageProjection::cloudHandler, this);

    // 原始点云帧
    pubFullCloud =
        pnh.advertise<sensor_msgs::PointCloud2>("/full_cloud_projected", 1);
    // 用于记录点云帧中点所属ring
    pubFullInfoCloud =
        pnh.advertise<sensor_msgs::PointCloud2>("/full_cloud_info", 1);

    // 属于地面的点云
    pubGroundCloud =
        pnh.advertise<sensor_msgs::PointCloud2>("/ground_cloud", 1);
    // 点云聚类分割
    pubSegmentedCloud =
        pnh.advertise<sensor_msgs::PointCloud2>("/segmented_cloud", 1);
    pubSegmentedCloudPure =
        pnh.advertise<sensor_msgs::PointCloud2>("/segmented_cloud_pure", 1);
    pubSegmentedCloudInfo =
        pnh.advertise<cloud_msgs::cloud_info>("/segmented_cloud_info", 1);
    // 聚类分割后的散点
    pubOutlierCloud =
        pnh.advertise<sensor_msgs::PointCloud2>("/outlier_cloud", 1);

    nanPoint.x = std::numeric_limits<float>::quiet_NaN();
    nanPoint.y = std::numeric_limits<float>::quiet_NaN();
    nanPoint.z = std::numeric_limits<float>::quiet_NaN();
    nanPoint.intensity = -1;

    // 开辟一些内存空间
    allocateMemory();
    resetParameters();
  }

  void allocateMemory() {
    laserCloudIn.reset(new pcl::PointCloud<PointType>());

    fullCloud.reset(new pcl::PointCloud<PointType>());
    fullInfoCloud.reset(new pcl::PointCloud<PointType>());

    groundCloud.reset(new pcl::PointCloud<PointType>());
    segmentedCloud.reset(new pcl::PointCloud<PointType>());
    segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
    outlierCloud.reset(new pcl::PointCloud<PointType>());

    laserCloudFullResColor.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

    fullCloud->points.resize(LINE_NUM * SCAN_NUM);
    fullInfoCloud->points.resize(LINE_NUM * SCAN_NUM);

    segMsg.startRingIndex.assign(LINE_NUM, 0);
    segMsg.endRingIndex.assign(LINE_NUM, 0);

    segMsg.segmentedCloudGroundFlag.assign(LINE_NUM * SCAN_NUM, false);
    segMsg.segmentedCloudColInd.assign(LINE_NUM * SCAN_NUM, 0);
    segMsg.segmentedCloudRange.assign(LINE_NUM * SCAN_NUM, 0);

    std::pair<int8_t, int8_t> neighbor;
    neighbor.first = -1;
    neighbor.second = 0;
    neighborIterator.push_back(neighbor);
    neighbor.first = 0;
    neighbor.second = 1;
    neighborIterator.push_back(neighbor);
    neighbor.first = 0;
    neighbor.second = -1;
    neighborIterator.push_back(neighbor);
    neighbor.first = 1;
    neighbor.second = 0;
    neighborIterator.push_back(neighbor);

    allPushedIndX = new uint16_t[LINE_NUM * SCAN_NUM];
    allPushedIndY = new uint16_t[LINE_NUM * SCAN_NUM];

    queueIndX = new uint16_t[LINE_NUM * SCAN_NUM];
    queueIndY = new uint16_t[LINE_NUM * SCAN_NUM];
  }

  void resetParameters() {
    laserCloudIn->clear();
    groundCloud->clear();
    segmentedCloud->clear();
    segmentedCloudPure->clear();
    outlierCloud->clear();

    rangeMat = cv::Mat(LINE_NUM, SCAN_NUM, CV_32F, cv::Scalar::all(FLT_MAX));
    groundMat = cv::Mat(LINE_NUM, SCAN_NUM, CV_8S, cv::Scalar::all(0));
    labelMat = cv::Mat(LINE_NUM, SCAN_NUM, CV_32S, cv::Scalar::all(0));
    labelCount = 1;

    std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
    std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(),
              nanPoint);
  }

  ~ImageProjection() {}

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

  // 从pcl_msg中提取出pcl
  void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg) {
    // 存储msg的时间戳和连接的frame
    cloudHeader = laserCloudMsg->header;
    // 提取pcl数据到laserCloudIn
    pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);
    std::vector<int> indices;
    // 去除nan点
    pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);
  }

  void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg) {
    // 计时
    TicToc ts_total;
    // 从pcl_msg中提取出pcl
    copyPointCloud(laserCloudMsg);
    // 求取该帧点云的起始点和终止点的角度,同时记录它们的差值
    findStartEndAngle();
    // 调整点云中点的索引,并组建Mat图像
    projectPointCloud();
    // 地面分割
    groundRemoval();
    // 聚类分割
    cloudSegmentation();
    // 发布相关点云数据
    publishCloud();
    resetParameters();
    double time_total = ts_total.toc();
  }

  void findStartEndAngle() {
    // 提取本帧点云第一个点的角度
    //lidar scan开始点的旋转角,atan2范围[-pi,+pi],计算旋转角时取负号是因为velodyne是顺时针旋转
    //将角度取负值相当于将逆时针转换成顺时针运动
    segMsg.startOrientation =
        -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);
    // 提取本帧点云最后一个点的角度
    segMsg.endOrientation =
        -atan2(laserCloudIn->points[laserCloudIn->points.size() - 1].y,
               laserCloudIn->points[laserCloudIn->points.size() - 2].x) +
        2 * M_PI; //lidar scan结束点的旋转角,因为atan2范围[-pi,+pi],加2*pi使点云旋转周期为2*pi
    // 要控制一帧点云的扫描范围要在合理范围
    if (segMsg.endOrientation - segMsg.startOrientation > 3 * M_PI) {
      segMsg.endOrientation -= 2 * M_PI;
    } else if (segMsg.endOrientation - segMsg.startOrientation < M_PI)
      segMsg.endOrientation += 2 * M_PI;
    // 该帧点云起始点与终止点间的角度,一般在360°左右
    segMsg.orientationDiff = segMsg.endOrientation - segMsg.startOrientation;
  }

  void projectPointCloud() {
    float verticalAngle, horizonAngle, range;
    size_t rowIdn, columnIdn, index, cloudSize;
    PointType thisPoint;

    cloudSize = laserCloudIn->points.size();

    // 遍历每个点
    for (size_t i = 0; i < cloudSize; ++i) {
      // copy该点用于操作
      thisPoint.x = laserCloudIn->points[i].x;
      thisPoint.y = laserCloudIn->points[i].y;
      thisPoint.z = laserCloudIn->points[i].z;

      // 该点垂直方向上的角度
      verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x +
                                              thisPoint.y * thisPoint.y)) *
                      180 / M_PI;
      // 该点属于的线数
      // 从下往上计数，-15度记为初始线，第0线，一共16线(N_SCAN=16)
      rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
      if (rowIdn < 0 || rowIdn >= LINE_NUM) continue;

      // 该点水平方向上的角度,注意这里求了atan(x/y)
      //  ^ y
      //  |     . thispoint
      //  |h_A /     
      //  |   /
      //  |^^/
      //  |-------------->x
      horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

      // round函数进行四舍五入取整
      // 这个操作使得雷达扫描由x轴旋转一周后点的水平角度为[0, 360°], 而不是[-180°, 180°],这里仍是逆时针旋转
      // 后面加减SCAN_NUM相关值的目的是为了调整每个点的索引,用于构建Mat image,不能存在负索引,而且0°的点要在图像中间
      columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + SCAN_NUM / 2;
      if (columnIdn >= SCAN_NUM) columnIdn -= SCAN_NUM;

      if (columnIdn < 0 || columnIdn >= SCAN_NUM) continue;

      // 该点测距值
      range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y +
                   thisPoint.z * thisPoint.z);
      rangeMat.at<float>(rowIdn, columnIdn) = range;

      // 存储相应索引在intensity中
      thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

      // 该点索引
      index = columnIdn + rowIdn * SCAN_NUM;
      fullCloud->points[index] = thisPoint;

      // InfoCloud额外用于存储点的距离值
      fullInfoCloud->points[index].intensity = range;
    }
  }

  void groundRemoval() {
    size_t lowerInd, upperInd;
    float diffX, diffY, diffZ, angle;

    for (size_t j = 0; j < SCAN_NUM; ++j) {
      // groundScanInd=5,即假设地面点只会存在与底下5条线上
      for (size_t i = 0; i < groundScanInd; ++i) {
        // 同一水平角度上相邻线的两个点
        lowerInd = j + (i)*SCAN_NUM;
        upperInd = j + (i + 1) * SCAN_NUM;

        // 存在无效点则跳出
        if (fullCloud->points[lowerInd].intensity == -1 ||
            fullCloud->points[upperInd].intensity == -1) {
          groundMat.at<int8_t>(i, j) = -1;
          continue;
        }

        diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
        diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
        diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

        angle = atan2(diffZ, sqrt(diffX * diffX + diffY * diffY)) * 180 / M_PI;

        // sensorMountAngle是雷达安装倾角,一般水平放置,因此是0°
        // angle足够小的话说明两点处于同一水平面附近,然后标记他们的聚类属性为1-->ground
        if (abs(angle - sensorMountAngle) <= 10) {
          groundMat.at<int8_t>(i, j) = 1;
          groundMat.at<int8_t>(i + 1, j) = 1;
        }
      }
    }

    // 遍历每条线
    for (size_t i = 0; i < LINE_NUM; ++i) {
      // 遍历线上每个点
      for (size_t j = 0; j < SCAN_NUM; ++j) {
        if (groundMat.at<int8_t>(i, j) == 1 ||
            rangeMat.at<float>(i, j) == FLT_MAX) { // FLT_MAX是最大的float数
          labelMat.at<int>(i, j) = -1; // 表示该点已被分类,不再参与后续聚类分割
        }
      }
    }
    // 存储地面点点云
    if (pubGroundCloud.getNumSubscribers() != 0) {
      for (size_t i = 0; i <= groundScanInd; ++i) {
        for (size_t j = 0; j < SCAN_NUM; ++j) {
          if (groundMat.at<int8_t>(i, j) == 1)
            groundCloud->push_back(fullCloud->points[j + i * SCAN_NUM]);
        }
      }
    }
  }

  void cloudSegmentation() {
    for (size_t i = 0; i < LINE_NUM; ++i)
      for (size_t j = 0; j < SCAN_NUM; ++j)
        // 对还没归类的点进行聚类分割
        if (labelMat.at<int>(i, j) == 0) labelComponents(i, j);

    int sizeOfSegCloud = 0;
    // 遍历每条线
    for (size_t i = 0; i < LINE_NUM; ++i) {
      // segMsg.startRingIndex[i]
			// segMsg.endRingIndex[i]
			// 表示第i线的点云起始序列和终止序列
			// 开始4点和末尾6点舍去不要
      segMsg.startRingIndex[i] = sizeOfSegCloud - 1 + 5;

      // 遍历线上每个点
      for (size_t j = 0; j < SCAN_NUM; ++j) {
        if (labelMat.at<int>(i, j) > 0 || groundMat.at<int8_t>(i, j) == 1) {
          // 经过聚类分割后标记为外点的点
          // labelMat数值为999999表示这个点是因为聚类数量不够30而被舍弃的点
					// 需要舍弃的点直接continue跳过本次循环，
					// 当列数为5的倍数，并且行数较大，可以认为非地面点的，将它保存进异常点云(界外点云)中
					// 然后再跳过本次循环
          if (labelMat.at<int>(i, j) == 999999) {
            if (i > groundScanInd && j % 5 == 0) {
              outlierCloud->push_back(fullCloud->points[j + i * SCAN_NUM]);
              continue;
            } else {
              continue;
            }
          }
          // 如果是地面点,对于列数不为5的倍数的，直接跳过不处理,相当于降采样的效果
          if (groundMat.at<int8_t>(i, j) == 1) {
            if (j % 5 != 0 && j > 5 && j < SCAN_NUM - 5) continue;
          }
          // 上面多个if语句已经去掉了不符合条件的点，这部分直接进行信息的拷贝和保存操作
					// 保存完毕后sizeOfSegCloud递增
          segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] =
              (groundMat.at<int8_t>(i, j) == 1);
          segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;
          segMsg.segmentedCloudRange[sizeOfSegCloud] = rangeMat.at<float>(i, j);
          segmentedCloud->push_back(fullCloud->points[j + i * SCAN_NUM]);
          ++sizeOfSegCloud;
        }
      }

      segMsg.endRingIndex[i] = sizeOfSegCloud - 1 - 5;
    }

    // 如果有节点订阅SegmentedCloudPure,
		// 那么把点云数据保存到segmentedCloudPure中去
    if (pubSegmentedCloudPure.getNumSubscribers() != 0) {
      for (size_t i = 0; i < LINE_NUM; ++i) {
        for (size_t j = 0; j < SCAN_NUM; ++j) {
          // 需要选择不是地面点(labelMat[i][j]!=-1)和没被舍弃的点
          if (labelMat.at<int>(i, j) > 0 && labelMat.at<int>(i, j) != 999999) {
            segmentedCloudPure->push_back(fullCloud->points[j + i * SCAN_NUM]);
            segmentedCloudPure->points.back().intensity =
                labelMat.at<int>(i, j);
          }
        }
      }
    }
  }

  void labelComponents(int row, int col) {
    float d1, d2, alpha, angle;
    int fromIndX, fromIndY, thisIndX, thisIndY;
    bool lineCountFlag[LINE_NUM] = {false};

    queueIndX[0] = row;
    queueIndY[0] = col;
    int queueSize = 1;
    int queueStartInd = 0;
    int queueEndInd = 1;

    allPushedIndX[0] = row;
    allPushedIndY[0] = col;
    int allPushedIndSize = 1;

    while (queueSize > 0) {
      fromIndX = queueIndX[queueStartInd];
      fromIndY = queueIndY[queueStartInd];
      --queueSize;
      ++queueStartInd;
      // labelCount的初始值为1，后面会递增
      labelMat.at<int>(fromIndX, fromIndY) = labelCount;

      // 四叉树的启发性最优搜索
      // neighbor=[[-1,0];[0,1];[0,-1];[1,0]]
			// 遍历点[fromIndX,fromIndY]边上的四个邻点
      for (auto iter = neighborIterator.begin(); iter != neighborIterator.end();
           ++iter) {
        thisIndX = fromIndX + (*iter).first;
        thisIndY = fromIndY + (*iter).second;

        // 图像以外则跳过
        if (thisIndX < 0 || thisIndX >= LINE_NUM) continue;

        // 首尾相连,形成圆柱形图像
        if (thisIndY < 0) thisIndY = SCAN_NUM - 1;
        if (thisIndY >= SCAN_NUM) thisIndY = 0;

        // 已经标记过该点,跳过
        if (labelMat.at<int>(thisIndX, thisIndY) != 0) continue;

        // 从深度图中取出两个相邻点的深度
        d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY),
                      rangeMat.at<float>(thisIndX, thisIndY));
        d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY),
                      rangeMat.at<float>(thisIndX, thisIndY));

        // alpha代表角度分辨率，
				// X方向上角度分辨率是segmentAlphaX(rad)
				// Y方向上角度分辨率是segmentAlphaY(rad)
        if ((*iter).first == 0)
          alpha = segmentAlphaX;
        else
          alpha = segmentAlphaY;

        // 计算论文中的\beta
        angle = atan2(d2 * sin(alpha), (d1 - d2 * cos(alpha)));

        // angle大于阈值则认为两点属于同一聚类
        if (angle > segmentTheta) {
          queueIndX[queueEndInd] = thisIndX;
          queueIndY[queueEndInd] = thisIndY;
          ++queueSize;
          ++queueEndInd;

          // 标记该点
          labelMat.at<int>(thisIndX, thisIndY) = labelCount;
          lineCountFlag[thisIndX] = true;

          allPushedIndX[allPushedIndSize] = thisIndX;
          allPushedIndY[allPushedIndSize] = thisIndY;
          ++allPushedIndSize;
        }
      }
    }

    bool feasibleSegment = false;
    // 如果聚类超过30个点，直接标记为一个可用聚类，labelCount需要递增
    if (allPushedIndSize >= 30)
      feasibleSegment = true;
    // 如果聚类点数小于30大于等于5，统计竖直方向上的聚类点数
    else if (allPushedIndSize >= segmentValidPointNum) {
      int lineCount = 0;
      for (size_t i = 0; i < LINE_NUM; ++i)
        if (lineCountFlag[i] == true) ++lineCount;
      if (lineCount >= segmentValidLineNum) feasibleSegment = true;
    }

    // 该聚类有效
    if (feasibleSegment == true) {
      ++labelCount;
    } else {
      for (size_t i = 0; i < allPushedIndSize; ++i) {
        labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
      }
    }
  }

  void publishCloud() {
    segMsg.header = cloudHeader;
    pubSegmentedCloudInfo.publish(segMsg);

    sensor_msgs::PointCloud2 laserCloudTemp;

    pcl::toROSMsg(*outlierCloud, laserCloudTemp);
    laserCloudTemp.header.stamp = cloudHeader.stamp;
    laserCloudTemp.header.frame_id = "vehicle";
    pubOutlierCloud.publish(laserCloudTemp);

    pcl::toROSMsg(*segmentedCloud, laserCloudTemp);
    laserCloudTemp.header.stamp = cloudHeader.stamp;
    laserCloudTemp.header.frame_id = "vehicle";
    pubSegmentedCloud.publish(laserCloudTemp);

    if (pubFullCloud.getNumSubscribers() != 0) {

      laserCloudFullResColor->clear();
      int laserCloudFullResNum = laserCloudIn->points.size();
      for (int i = 0; i < laserCloudFullResNum; i++) {
        pcl::PointXYZRGB temp_point;
        RGBpointAssociateToMap(&laserCloudIn->points[i], &temp_point);
        laserCloudFullResColor->push_back(temp_point);
      }

      // pcl::toROSMsg(*fullCloud, laserCloudTemp);
      pcl::toROSMsg(*laserCloudFullResColor, laserCloudTemp);
      laserCloudTemp.header.stamp = cloudHeader.stamp;
      laserCloudTemp.header.frame_id = "vehicle";
      pubFullCloud.publish(laserCloudTemp);
    }

    if (pubGroundCloud.getNumSubscribers() != 0) {
      pcl::toROSMsg(*groundCloud, laserCloudTemp);
      laserCloudTemp.header.stamp = cloudHeader.stamp;
      laserCloudTemp.header.frame_id = "vehicle";
      pubGroundCloud.publish(laserCloudTemp);
    }

    if (pubSegmentedCloudPure.getNumSubscribers() != 0) {
      pcl::toROSMsg(*segmentedCloudPure, laserCloudTemp);
      laserCloudTemp.header.stamp = cloudHeader.stamp;
      laserCloudTemp.header.frame_id = "vehicle";
      pubSegmentedCloudPure.publish(laserCloudTemp);
    }

    if (pubFullInfoCloud.getNumSubscribers() != 0) {
      pcl::toROSMsg(*fullInfoCloud, laserCloudTemp);
      laserCloudTemp.header.stamp = cloudHeader.stamp;
      laserCloudTemp.header.frame_id = "vehicle";
      pubFullInfoCloud.publish(laserCloudTemp);
    }
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "image_projection_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");


  parameter::readParameters(pnh);


  ImageProjection featureHandler(nh, pnh);

  ROS_INFO("\033[1;32m---->\033[0m Feature Extraction Module Started.");

  ros::spin();
  return 0;
}
