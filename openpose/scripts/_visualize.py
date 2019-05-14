#! /usr/bin/env python

import rospy
from openpose_ros.srv import Compute
from openpose_ros.msg import SparseTensor
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
from dynamic_reconfigure.client import Client as ReconfClient
from _openpose import encode_sparse_tensor, decode_sparse_tensor
from nets import non_maximum_suppression, connect_parts
from labels import POSE_BODY_25_L2, POSE_BODY_25_L1

bridge = CvBridge()
image = None
limbs = [(POSE_BODY_25_L2.index(p), POSE_BODY_25_L2.index(q)) \
         for p, q in POSE_BODY_25_L1]

def sizeof_sparse_tensor(msg):
    size = 14 #width to max_value
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
    image_sp = SparseTensor()
    image_sp.name = 'image'
    image_sp.width = cv_image.shape[1]
    image_sp.height = cv_image.shape[0]
    image_sp.channels = cv_image.shape[2]
    image_sp.min_value = 0.
    image_sp.max_value = 1.
    image_sp.quantized_values = cv_image[:,:,::-1].reshape(-1).tolist()

    t0 = rospy.Time.now()
    msg1 = compute_openpose_tx2(
    #msg1 = compute_openpose_xavier(
        input_tensors=[image_sp],
        queries=['stage1_L1', 'stage0'],
        thresholds=[0.1, 0.1])
    #affinity = decode_sparse_tensor(hidden_msg.affinity_field)
    #affinity = np.sqrt(affinity[:,:,50]**2 + affinity[:,:,51]**2)
    #affinity = np.stack([np.zeros_like(affinity)]*2+[affinity], 2)
    #affinity = cv2.resize(affinity, (640,480))
    t1 = rospy.Time.now()
    msg2 = compute_openpose_xavier(
        input_tensors=msg1.output_tensors,
        queries=['stage4_L1', 'stage2_L2'],
        thresholds=[0.1, 0.1])
    t2 = rospy.Time.now()
    affinity = decode_sparse_tensor(msg2.output_tensors[0])
    heat_map = decode_sparse_tensor(msg2.output_tensors[1])
    keypoints = non_maximum_suppression(
        np.expand_dims(heat_map[:,:,:-1], 0), threshold=0.15)
    sx = image_msg.width/float(heat_map.shape[1])
    sy = image_msg.height/float(heat_map.shape[0])
    persons = connect_parts(affinity, keypoints[:,1:], limbs,
                            line_division=15, threshold=0.08)
    persons = [ { k: (int(keypoints[v][2]*sx+sx/2),
                      int(keypoints[v][1]*sy+sy/2)) \
                  for k,v in p.items() } for p in persons ]
    t3 = rospy.Time.now()

    sizes = list(map(sizeof_sparse_tensor, msg1.output_tensors))
    print("Sizes: {} bytes\nTotal: {} bytes".format(sizes, sum(sizes)))
    print("Image to stage{0}: {1} sec\nStage{0} to detection: {2} sec\nPost processing: {3} sec\nTotal: {4} sec" \
          .format(stage, (t1-t0).to_sec(), (t2-t1).to_sec(),
                  (t3-t2).to_sec(), (t3-t0).to_sec()))
    print("Found {} people.".format(len(persons)))
    
    for person in persons:
        for k, v in person.items():
            cv2.circle(cv_image, v, 3, (0, 0, 255), 1)
        for p1, p2 in limbs:
            if p1 in person and p2 in person:
                cv2.line(cv_image,
                         person[p1], person[p2],
                         (0, 255, 0), 2)
    
    global image
    image = cv_image

if __name__ == '__main__':
    rospy.init_node('visualize_openpose')
    stage = rospy.get_param('~stage', 1)
    
    compute_openpose_tx2 = rospy.ServiceProxy('compute_openpose_tx2-1', Compute)
    #compute_openpose_tx2.wait_for_service()
    compute_openpose_xavier = rospy.ServiceProxy('compute_openpose_xavier-1', Compute)
    compute_openpose_xavier.wait_for_service()

    reconf = ReconfClient('openpose_xavier-1')
    reconf.update_configuration({'affinity_threshold': 0.1})

    image_sub = rospy.Subscriber('image', Image, callback)
    while not rospy.is_shutdown():
        if image is not None:
            cv2.imshow('OpenPose visualization', image)
        cv2.waitKey(1)
