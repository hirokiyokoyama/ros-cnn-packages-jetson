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

def sizeof_sparse_tensor(msg):
    size = 11 #width to max_value
    size += len(msg.x_indices)
    size += len(msg.y_indices)
    size += len(msg.channel_indices)
    size += len(msg.quantized_values)
    return size

def callback(image_msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
    except CvBridgeError as e:
        rospy.logerr(e)

    t0 = rospy.Time.now()
    hidden_msg = compute_openpose_tx2(
        input='image', output='stage%d' % stage,
        image=image_msg)
    t1 = rospy.Time.now()
    people_msg = compute_openpose_xavier(
        input='stage%d' % stage, output='people',
        feature_map=hidden_msg.feature_map,
        affinity_field=hidden_msg.affinity_field,
        confidence_map=hidden_msg.confidence_map)
    t2 = rospy.Time.now()

    size1 = sizeof_sparse_tensor(hidden_msg.feature_map)
    size2 = sizeof_sparse_tensor(hidden_msg.affinity_field)
    size3 = sizeof_sparse_tensor(hidden_msg.confidence_map)
    print("Feature map: {} bytes\n Affinity field: {} bytes\nConfidence map: {} bytes\nTotal: {} bytes" \
          .format(size1, size2, size3, size1+size2+size3))
    print("Image to stage{0}: {1} sec\nStage{0} to detection: {2} sec\nTotal: {3} sec" \
          .format(stage, (t1-t0).to_sec(), (t2-t1).to_sec(), (t2-t0).to_sec()))
    
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
    stage = rospy.get_param('~stage', 1)
    
    compute_openpose_tx2 = rospy.ServiceProxy('compute_openpose_tx2-1', Compute)
    compute_openpose_tx2.wait_for_service()
    compute_openpose_xavier = rospy.ServiceProxy('compute_openpose_xavier-1', Compute)
    compute_openpose_xavier.wait_for_service()
    image_sub = rospy.Subscriber('image', Image, callback)
    while not rospy.is_shutdown():
        if image is not None:
            cv2.imshow('OpenPose visualization', image)
        cv2.waitKey(1)