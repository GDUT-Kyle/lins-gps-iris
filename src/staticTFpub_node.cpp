#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <string>
#include <sleipnir_msgs/sensorgps.h>
#include "parameters.h"
#include "xmldom/XmlDomDocument.h"

using namespace std;
using namespace parameter;

class staticTFpub
{
public:
    ros::NodeHandle pnh;
    ros::Subscriber imu_sub;
    tf::TransformBroadcaster pub_map_tf_pcl_map;
    tf::Transform map_tf_pcl_map;

    tf::TransformBroadcaster pub_map_tf_planning_map;
    tf::Transform map_tf_planning_map;

    string osm_file;

    shared_ptr<XmlDomDocument> doc;

    double mgrs_x = 0;
    double mgrs_y = 0;
    string xml_value;
public:
    staticTFpub(ros::NodeHandle& nh) : pnh(nh) {
        imu_sub = pnh.subscribe(IMU_TOPIC, 100, &staticTFpub::ImuHandler, this);

        map_tf_pcl_map.setOrigin(tf::Vector3(InitPose_x, InitPose_y, 0.0));
        tf::Quaternion q;
        q.setRPY(0, 0, InitPose_yaw);
        map_tf_pcl_map.setRotation(q);

        pnh.param<string>("osm_file", osm_file, "osm.yaml");

        doc = make_shared<XmlDomDocument>(osm_file.c_str());

        if(doc)
        {
            for(int i=0; i<doc->getChildCount("node", 0, "tag"); i++)
            {
                if(doc->getChildAttribute("node", 0, "tag", i, "k") == "local_x")
                {
                    mgrs_x = atof(doc->getChildAttribute("node", 0, "tag", i, "v").c_str());
                }
                if(doc->getChildAttribute("node", 0, "tag", i, "k") == "local_y")
                {
                    mgrs_y = atof(doc->getChildAttribute("node", 0, "tag", i, "v").c_str());
                }
            }
        }

        map_tf_planning_map.setOrigin(tf::Vector3(mgrs_x, mgrs_y, 0.0));
        q.setRPY(0, 0, 0);
        map_tf_planning_map.setRotation(q);
    }
    void ImuHandler(const sensor_msgs::Imu::ConstPtr& msg)
    {
        pub_map2pcl_map(msg->header.stamp);
        pub_map2planning_map(msg->header.stamp);
    }
    void pub_map2pcl_map(ros::Time timestamp)
    {
        pub_map_tf_pcl_map.sendTransform(tf::StampedTransform(map_tf_pcl_map, timestamp, "map", "pcl_map"));
    }
    void pub_map2planning_map(ros::Time timestamp)
    {
        pub_map_tf_planning_map.sendTransform(tf::StampedTransform(map_tf_planning_map, timestamp, "map", "planning_map"));
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "staticTFpub_node");
    ros::NodeHandle nh("~");

    parameter::readInitPose(nh);

    staticTFpub staticTFpub_(nh);

    ros::Rate rate(500); // 10Hz

    while(ros::ok())
    {
        rate.sleep();
        ros::spinOnce();
        // staticTFpub_.pub_map2pcl_map();
        // staticTFpub_.pub_map2planning_map();
    }

    return 0;
}