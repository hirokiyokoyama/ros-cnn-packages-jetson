#! /usr/bin/env python

import rospy
from openpose_ros.srv import Compute
from openpose_ros.msg import SparseTensor
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
from labels import POSE_COCO_L1

from openpose import encode_sparse_tensor

bridge = CvBridge()
image = None

def callback(image_msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
    except CvBridgeError as e:
        rospy.logerr(e)

    hidden_msg = compute(input='image',
                         output='stage1',
                         image=image_msg)
    people_msg = compute(input='stage1',
                         output='people',
                         feature_map=hidden_msg.feature_map,
                         affinity_field=hidden_msg.affinity_field,
                         confidence_map=hidden_msg.confidence_map)
    for person in people_msg.people:
        for part in person.body_parts:
            cv2.circle(cv_image, (int(part.x), int(part.y)), 3, (0, 0, 255), 1)
        parts = {part.name: (int(part.x), int(part.y)) for part in person.body_parts}
        for p1, p2 in POSE_COCO_L1:
            if p1 in parts and p2 in parts:
                cv2.line(cv_image, parts[p1], parts[p2], (0, 255, 0), 2)
    
    global image
    image = cv_image

if __name__ == '__main__':
    rospy.init_node('visualize_openpose')
    compute = rospy.ServiceProxy('compute_openpose', Compute)
    compute.wait_for_service()
    image_sub = rospy.Subscriber('image', Image, callback)
    while not rospy.is_shutdown():
        if image is not None:
            cv2.imshow('OpenPose visualization', image)
        cv2.waitKey(1)
