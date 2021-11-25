### 修改日志

2020/10/02

添加sensorgps的数据到mapping程序中，无清洗添加至因子图进行融合，经测试通过√

2020/10/02

由于我们使用的GPS组合导航模块包含姿态测量（发现咱们这玩意三个旋转轴的旋转方向都与笛卡尔直角坐标系相反，这个得注意），因此我们不适用GTSAM的GPS factor，而是采用pose3 factor。
回环矫正方面，发现sc-lego-loam的sc检测后添加的回环约束因子是当前帧与建图第一帧的约束关系，这不合理，应该是当前帧与闭环匹配帧，已修改
因子图的噪声方面感觉设置得不合理，需要后续的实验和调节。

2020/10/03
添加了纯定位模块，其中出现了问题：

> 问题描述：在做localization时发现地图栅格分配不足，我们直接拿了LOAM的栅格数，那个是对应小尺度地图的，所以我们现在要把`laserCloudNum`修改得足够大。

问题已解决，将地图总栅格数修改大一些：

```c++
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
```

已经测试

2020/10/04

了解了use_sim_time的作用，我们的rviz指定初始位置也需要一个时钟源，因此如果要调试rviz设置情况的话，play rosbag的时候应该只播放时钟：

```shell
rosbag play 2020-09-28-20-46-00.bag --clock --topics
```

明天设定一下map和camera_init的方向情况，然后尝试使用rviz设定`transformTobeMapped`的值



2020/10/06

尝试修改复杂的`transformAssociateToMap()`函数，使用`Eigen`的`Quaternion`进行简化运算

该函数的作用：计算连续两个Lidar odometry数据（`transformBefMapped`是上一次的里程计，`transformSum`是本次的里程计）间的位姿变换，然后将这个里程计估计出的帧间位姿变换附加到`transformTobeMapped`，然后作为优化的初值。

修改思路：`transformTobeMapped`,`transformBefMapped`和`transformSum`都转成`Quaternion`或齐次变换矩阵形式，然后进行姿态运算，最后再进行逆变换

修改成功，新的函数为`transformAssociateToMap_()`，但是发现一个问题，从四元数转欧拉角时得到的数值跟原函数求得的数值不一致，但是组合起来的旋转作用又是一致的，还需要进一步探索。保守起见还是用回原来的函数。



2020/10/07

Localization module已添加上手动（rviz）指定初值，主要是手动修改`transformAftMapped`，注意不是`transformTobeMapped`，具体原因可以溯回到`transformAssociateToMap()`，`transformTobeMapped`是基于`transformAftMapped`的



2020/11/01

在mapping的launch文件中添加了保存地图的选项，这样不用每次都保存地图浪费调试时间



2020/11/05

修正了Localization 和mapping module中激光雷达和IMU的topic名固定为`velodyne_points`的问题，现在改成了每次都从config文件中读取，这样比较灵活

重新加入了全局地图发布线程，每10秒发布一次，方便测试

在`transform_fusion_node`中加了转WGS84的function，可以发布出来，但是与原始的GPS数据一起放在openstreet中，发现偏差较大，需要查找原因



2020/11/11

修正了transform_fusion_node /map转WGS84时出现误差的问题（由于在launch文件中传递的动态参数与mapLocalizaiton有相同，也就是有冲突，因此将两个节点需要用到的参数改成launch全局参数，具体使用方法参考`花里胡哨笔记`）



2020/11/18

使用新的bag建图后，在进行localization时发现GPS的轨迹与Lidar定位出来的轨迹不一致，但是经过再三确认在建图时明明紧紧跟随着GPS的轨迹，非常迷惑

解决方法：原来我们在localization时的GPS起始点使用的是bag的第一个数据，但是我们在建图程序中选取的不是bag的第一个数据，而是与第一个keyframe时间戳匹配的GPS数据，因此我们下次再在新的map中定位时，必须选取第一个keyframe对应的GPS数据，而不是bag的第一个数据。



2020/11/28

根据浩源导航程序的需求，在`transform_fusion_node.cpp`中加入了**发布经过融合后的定位数据**，消息类型为`sleipnir_msgs/gps4control`，经过功能包测试可以使用，还没部署到车上。



2020/11/29

经过试车测试，从建筑物出来时GPS的跳变影响比较大，需要设置一个缓冲队列，然后还可以适当增加激光雷达的置信度，还有个想法是当从建筑物出来时，需要设置一个GPS的信赖域，如果离当前定位数据太远的话就认为GPS还没恢复



2020/11/30

在实车测试时发现当车辆从建筑物出来后搜索到GPS信号的短暂时间内，GPS信号恢复时会发生突变，这时GPS的状态是良好的，但是他的定位数据还没恢复完成，这会将定位数据融合结果“扯“到很远，因此加了一个GPS的缓冲队列用于避免这种情况，而且还额外添加了判断当前位置与GPS定位的偏差程度，如果偏差太远的话也会认为GPS不正常。在融合Lidar因子时，加了一个选项可以选择使用一元因子还是传统的二元因子，一元因子的话噪声固定，可能会更加稳定。

2020/12/05

这一版调整了GPS切换策略，经过实车测试可行，但是还未经浩源的轨迹跟踪程序测试。

在测试过程中发现，车辆经过室内外交界处时会发生定位的轻微跳变，应该是因为此时GPS失效，只有Odometry因子起作用，这需要再融一个传感器信息去约束这种跳变。感觉还需要加上约束因素，因此我打算将IMU预积分项作为约束因子，添加到后端中，也就是二次融合IMU数据，从而起到平滑的作用。为了方便使用GTSAM中的IMU预积分因子，我将`lidar_mapping_node.cpp`和`mapLocalization.cpp`中的GTSAM因子标签改成了`Symbol`形式，位姿存在**X(a)**中，其中要注意的是，**a**需要>=0。



2020/12/12

我在建图程序中加了IMU预积分因子，测试通过，但是感觉改进效果不明显，有待挖掘

在Localization里加上IMU预积分因子时还不太行，我觉得原因是因为我们程序中设了初始位姿为0，但是GPS因子一开始将位姿扯到远处，导致因子图中的速度状态飞了，明天再想想怎么改，先下班了。



2020/12/13

localization程序中，当车辆停在原地或者运动幅度很小时会一直抖动，问题应该出在了融合方面，当车辆行驶时会生成Keypoint，就会被GPS约束在某个点，但是停在原地或者运动幅度很小时不会运行`saveKeyFramesAndFactor`，这时输入到`transform_fusion_node`的`aftMappedTrans`没有融合GPS而是完全由点云匹配生成，但是呢上一个被GPS约束后的点不是点云匹配最佳的位置，因此会被拉扯到点云匹配最佳处，因此造成了位置反复横跳。

已解决：LEGO-LOAM建图是需要回环，为了避免关键帧过多，他通过距离对关键点进行稀疏化，但是在localization中不需要回环，因此将距离判断去掉，每一次后端优化都加入到GTSAM中进行融合，这样就不会反复横跳。但是有个疑问，长时间的积累，因子图会不会内存爆了？需要设置个滑动窗口吗？

答：不需要，知乎回答：[增量式smooth and mapping 是不是伪需求啊？](https://www.zhihu.com/question/279027808)



2021/03/11

在`transform_fusion_node`文件中，将定位结果转换成GPS经纬度信息发布出去。但是经过坐标系统一后，发现规划处获取得到的定位结果与我这显示的定位结果存在偏差，经过测试发现是存在以建图原点为圆心的微量旋转（如0.003 rad），明明数据来源都是来自定位这里，但是为什么会不一致呢？目前还没找到原因（很大可能是来自传感器外参），因此我这里加了一个可调节参数用来调整该参数，如代码：

```cpp
	// get the GPS origination
    // 我们车上使用的星网宇达GPS接收器，他的航向角方向与笛卡尔直角坐标系相反
    Eigen::Vector3d  InitEulerAngle=Eigen::Vector3d((-OriYaw+90.0)*deg+compensate_init_yaw+unknow_yaw_para, -OriPitch*deg, OriRoll*deg);
    cout<<"unknow_yaw_para: "<<unknow_yaw_para<<endl;
    Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(InitEulerAngle(2),Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(InitEulerAngle(1),Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(InitEulerAngle(0),Eigen::Vector3d::UnitZ()));
    // init_fix_odom_pose = yawAngle*pitchAngle*rollAngle;
    Eigen::Quaterniond init_fix_odom_pose;
    init_fix_odom_pose = yawAngle;
```

`unknow_yaw_para`就是用来调整的参数，单位为弧度，可以在launch文件中进行调节