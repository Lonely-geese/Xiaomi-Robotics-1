import os,time,json
import rclpy
rclpy.init();n=rclpy.create_node('inspect_head_camera_'+os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'].lower())
for _ in range(50):rclpy.spin_once(n,timeout_sec=.1)
info=[]
for topic in ['/head_depth_cam/color/image_raw/compressed','/left_hand_depth_cam/color/image_raw/compressed','/right_hand_depth_cam/color/image_raw/compressed']:
 publishers=n.get_publishers_info_by_topic(topic)
 info.append({'topic':topic,'publishers':[{'name':p.node_name,'namespace':p.node_namespace,'gid':bytes(p.endpoint_gid).hex()} for p in publishers]})
print(json.dumps({'discovery':os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'],'domain':os.environ.get('ROS_DOMAIN_ID','0'),'topics':info,'nodes':n.get_node_names_and_namespaces()},ensure_ascii=False))
n.destroy_node();rclpy.shutdown()
