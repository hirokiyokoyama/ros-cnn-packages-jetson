#! /usr/bin/env python

import rospy
from openpose_ros.srv import Compute
from openpose_ros.msg import PersonArray, Person, SparseTensor, SparseTensorArray
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
xavier_people = None
tx2_people = None
limbs = [(POSE_BODY_25_L2.index(p), POSE_BODY_25_L2.index(q)) \
         for p, q in POSE_BODY_25_L1]

def sizeof_sparse_tensor(msg):
  size = 14 #width to max_value
  size += len(msg.x_indices)
  size += len(msg.y_indices)
  size += len(msg.channel_indices)
  size += len(msg.quantized_values)
  return size

def extract_people(heat_map, affinity):
  keypoints = non_maximum_suppression(
      np.expand_dims(heat_map[:,:,:-1], 0), threshold=0.15)
  persons = connect_parts(affinity, keypoints[:,1:], limbs,
                          line_division=15, threshold=0.08)
  persons = [ { k: ((keypoints[v][2]+0.5)/heat_map.shape[1],
                    (keypoints[v][1]+0.5)/heat_map.shape[0]) \
                for k,v in p.items() } for p in persons ]
  return persons

def draw_people(image, people):
  for person in people:
    for k, v in person.items():
      v = (int(v[0]*image.shape[1]), int(v[1]*image.shape[0]))
      person[k] = v
      cv2.circle(image, v, 3, (0, 0, 255), 1)
    for p1, p2 in limbs:
      if p1 in person and p2 in person:
        cv2.line(image,
                 person[p1], person[p2],
                 (0, 255, 0), 2)
        
def draw_peoplemsg(image, people):
  for person in people.people:
    points = {}
    for part in person.body_parts:
      k = part.name
      v = (int(part.x*image.shape[1]), int(part.y*image.shape[0]))
      points[k] = v
      cv2.circle(image, v, 3, (0, 0, 255), 1)
    for p1, p2 in POSE_BODY_25_L1:
      if p1 in points and p2 in points:
        cv2.line(image,
                 points[p1], points[p2],
                 (0, 255, 0), 2)

def image_cb(image_msg):
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
    people = extract_people(heat_map, affinity)
    t3 = rospy.Time.now()
    
    sizes = list(map(sizeof_sparse_tensor, msg1.output_tensors))
    print("Sizes: {} bytes\nTotal: {} bytes".format(sizes, sum(sizes)))
    print("Image to stage{0}: {1} sec\nStage{0} to detection: {2} sec\nPost processing: {3} sec\nTotal: {4} sec" \
          .format(stage, (t1-t0).to_sec(), (t2-t1).to_sec(),
                  (t3-t2).to_sec(), (t3-t0).to_sec()))
    print("Found {} people.".format(len(people)))

    global xavier_image, tx2_image

    _xavier_image = cv_image.copy()
    draw_people(_xavier_image, people)
    xavier_image = _xavier_image
    
    #
    msg1.output_tensors[0].name = 'stage4_L1'
    msg3 = compute_openpose_xavier(
        input_tensors=msg1.output_tensors,
        queries=['stage1_L2'],
        thresholds=[0.1, 0.1])
    affinity = decode_sparse_tensor(msg1.output_tensors[0])
    heat_map = decode_sparse_tensor(msg3.output_tensors[0])
    people = extract_people(heat_map, affinity)
    _tx2_image = cv_image
    draw_people(_tx2_image, people)
    tx2_image = _tx2_image

def sp_cb(msg1):
    t0 = rospy.Time.now()
    msg2 = compute_openpose_xavier(
        input_tensors=msg1.sparse_tensors,
        queries=['stage4_L1', 'stage2_L2'],
        thresholds=[0.1, 0.1])
    t1 = rospy.Time.now()
    affinity = decode_sparse_tensor(msg2.output_tensors[0])
    heat_map = decode_sparse_tensor(msg2.output_tensors[1])
    people = extract_people(heat_map, affinity)
    t2 = rospy.Time.now()
    
    sizes = list(map(sizeof_sparse_tensor, msg1.sparse_tensors))
    print("Sizes: {} bytes\nTotal: {} bytes".format(sizes, sum(sizes)))
    print("Stage1 to detection: {} sec\nPost processing: {} sec\nTotal: {} sec" \
          .format((t1-t0).to_sec(), (t2-t1).to_sec(), (t2-t0).to_sec()))
    print("Found {} people.".format(len(people)))

    global xavier_image, tx2_image

    _xavier_image = np.zeros((480,640,3), dtype=np.uint8)
    draw_people(_xavier_image, people)
    xavier_image = _xavier_image
    
    """
    msg1.sparse_tensors[1].name = 'stage4_L1'
    msg3 = compute_openpose_xavier(
        input_tensors=msg1.sparse_tensors,
        queries=['stage1_L2'],
        thresholds=[0.1, 0.1])
    affinity = decode_sparse_tensor(msg1.sparse_tensors[0])
    heat_map = decode_sparse_tensor(msg3.output_tensors[0])
    people = extract_people(heat_map, affinity)
    _tx2_image = np.zeros((480,640,3), dtype=np.uint8)
    draw_people(_tx2_image, people)
    tx2_image = _tx2_image
    """
    
def img_cb(image_msg):
  global image
  try:
    image = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
  except CvBridgeError as e:
    rospy.logerr(e)
    
def people_cb1(msg):
    global xavier_people
    xavier_people = msg
    
def people_cb2(msg):
    global tx2_people
    tx2_people = msg

if __name__ == '__main__':
  rospy.init_node('visualize_openpose')
  stage = rospy.get_param('~stage', 1)
    
  compute_openpose_xavier = rospy.ServiceProxy('compute_openpose', Compute)
  compute_openpose_xavier.wait_for_service()

  reconf_xavier = ReconfClient('openpose_xavier')
  reconf_xavier.update_configuration({
    'key_point_threshold': 0.1,
    'affinity_threshold': 0.1,
    'line_division': 5
  })
  reconf_tx2 = ReconfClient('openpose_tx2')
  reconf_tx2.update_configuration({
    'key_point_threshold': 0.1,
    'affinity_threshold': 0.1,
    'line_division': 2
  })

  #st_sub = rospy.Subscriber('openpose_stage1', SparseTensorArray,
  #                          sp_cb, queue_size=1)
  rospy.Subscriber('image', Image, img_cb,
                   queue_size=1, buff_size=1048576*8, tcp_nodelay=True)
  rospy.Subscriber('people_xavier', PersonArray, people_cb1,
                   queue_size=1, buff_size=1048576*8, tcp_nodelay=True)
  rospy.Subscriber('people_tx2', PersonArray, people_cb2,
                   queue_size=1, buff_size=1048576*8, tcp_nodelay=True)
  while not rospy.is_shutdown():
    if image is not None:
      xavier_image = image.copy()
      tx2_image = image.copy()
      if xavier_people is not None:
        draw_peoplemsg(xavier_image, xavier_people)
      if tx2_people is not None:
        draw_peoplemsg(tx2_image, tx2_people)
      cv2.imshow('Xavier', xavier_image)
      cv2.imshow('TX2', tx2_image)
    cv2.waitKey(1)
