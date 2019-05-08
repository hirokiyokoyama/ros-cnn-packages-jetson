#! /usr/bin/env python

import rospy
from openpose_ros.srv import DetectKeyPoints
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from dynamic_reconfigure.client import Client as ReconfClient
import numpy as np

bridge = CvBridge()
image = None

def callback(image_msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
    except CvBridgeError as e:
        rospy.logerr(e)

    res = detect_face(image_msg)
    print(res.key_points)
    for kp in res.key_points:
        cv2.circle(cv_image, (int(kp.x), int(kp.y)), 3, (0, 0, 255), 1)
    
    global image
    image = cv_image

if __name__ == '__main__':
    rospy.init_node('visualize_openpose')
    detect_face = rospy.ServiceProxy('detect_face', DetectKeyPoints)
    detect_face.wait_for_service()
    reconf = ReconfClient('openpose')
    reconf.update_configuration({'key_point_threshold': 0.05})
    image_sub = rospy.Subscriber('image', Image, callback)
    while not rospy.is_shutdown():
        if image is not None:
            cv2.imshow('OpenPose visualization', image)
        cv2.waitKey(1)
