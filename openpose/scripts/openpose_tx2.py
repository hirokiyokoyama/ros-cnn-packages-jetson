#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import rospy
import rospkg
import os
from cv_bridge import CvBridge, CvBridgeError
from detector import KeyPointDetector, encode_sparse_tensor, decode_sparse_tensor

from sensor_msgs.msg import Image
from dynamic_reconfigure.server import Server as ReconfServer

from openpose_ros.msg import Person, PersonArray, KeyPoint
from openpose_ros.msg import SparseTensor, SparseTensorArray
from openpose_ros.srv import Compute, ComputeResponse
from openpose_ros.cfg import KeyPointDetectorConfig
from nets import connect_parts
from std_msgs.msg import Duration
from std_srvs.srv import Empty, EmptyResponse

def callback(data):
  publish_mid = mid_pub.get_num_connections() > 0
  publish_people = people_pub.get_num_connections() > 0
  if not publish_mid and not publish_people:
    return
  
  try:
    cv_image = bridge.imgmsg_to_cv2(data, 'rgb8')
  except CvBridgeError as e:
    rospy.logerr(e)
    return
  
  fetch_list = ['stage0_sparse', 'part_affinity_fields_sparse']
  feed_dict = {'image': [cv_image/255.]}
  if publish_people:
    fetch_list.append('key_points')
    feed_dict['key_point_threshold'] = pose_params['key_point_threshold']
  
  t0 = rospy.Time.now()
  outputs = pose_detector.compute(fetch_list, feed_dict)
  t1 = rospy.Time.now()
  rospy.loginfo('Image to mid-stage: {} sec'.format((t1-t0).to_sec()))

  if publish_mid:
    msg = SparseTensorArray()
    msg.header = data.header
    stage0 = sparse_tensor_value_to_msg(outputs[0])
    stage0.name = 'stage0'
    l1 = sparse_tensor_value_to_msg(outputs[1])
    l1.name = stage_n_L1
    msg.sparse_tensors = [stage0, l1]
    mid_pub.publish(msg)

    t2 = rospy.Time.now()
    rospy.loginfo('Publishing mid-stage msg: {} sec'.format((t2-t1).to_sec()))
    t1 = t2

  if publish_people:
    affinity = sparse_tensor_value_to_array(outputs[1])
    keypoints = outputs[2][:,1:]
    people = connect_parts(affinity[:,:,affinity_inds], keypoints, limbs,
                           line_division=pose_params['line_division'],
                           threshold=pose_params['affinity_threshold'])
    h, w = affinity.shape[:2]
    people = [ { pose_detector._part_names[k]: ((keypoints[v][1]+0.5)/w, (keypoints[v][0]+0.5)/h) \
               for k,v in person.items()} for person in people ]
    
    msg = PersonArray()
    msg.header = data.header
    msg.people = [Person(body_parts=[KeyPoint(name=k, x=x, y=y) \
                                     for k,(x,y) in p.items()]) \
                  for p in people]
    people_pub.publish(msg)

    t2 = rospy.Time.now()
    rospy.loginfo('Publishing people msg: {} sec'.format((t2-t1).to_sec()))
    t1 = t2
    
  time_pub.publish(t1-t0)
    
def compute(req):
  feed_dict = { x.name: [decode_sparse_tensor(x)] for x in req.input_tensors }
  outputs = pose_detector.compute(req.queries, feed_dict)
  thresholds = req.thresholds
  if not thresholds:
    thresholds = [0.1] * len(req.queries)
  output_tensors = []
  for o, t, q in zip(outputs, thresholds, req.queries):
    st = encode_sparse_tensor(o[0], threshold=t)
    st.name = q
    output_tensors.append(st)
  return ComputeResponse(output_tensors=output_tensors)

def reconf_callback(config, level):
  for key in ['key_point_threshold', 'affinity_threshold', 'line_division']:
    pose_params[key] = config[key]
  return config

def dense_to_sparse(x, threshold=0.1):
  x = tf.where(tf.abs(x)>0.1, x, tf.zeros_like(x))
  x = tf.contrib.layers.dense_to_sparse(x)
  return x

def sparse_tensor_value_to_msg(x):
  msg = SparseTensor()
  msg.height, msg.width, msg.channels = x.dense_shape.tolist()
  if len(x.values) > 0:
    min_val = x.values.min()
    max_val = x.values.max()
  else:
    min_val = 0.
    max_val = 1.
  msg.min_value = min_val
  msg.max_value = max_val
  msg.x_indices = x.indices[:,1].tolist()
  msg.y_indices = x.indices[:,0].tolist()
  msg.channel_indices = x.indices[:,2].tolist()
  msg.quantized_values = np.uint8((x.values-min_val)/(max_val-min_val)*255).tolist()
  return msg

def sparse_tensor_value_to_array(x):
  y = np.zeros(x.dense_shape)
  y[tuple(x.indices.T)] = x.values
  return y

def initialize_network(req=None):
  global stage_n_L1, stage_n_L2
  l1_stage = rospy.get_param('~l1_stage', 4)
  l2_stage = rospy.get_param('~l2_stage', 2)
  stage_n_L1 = 'stage%d_L1' % l1_stage
  stage_n_L2 = 'stage%d_L2' % l2_stage
  
  image_width = rospy.get_param('openpose_image_width', 360)
  image_height = rospy.get_param('openpose_image_height', 270)
  input_shape = (image_height, image_width)
  
  pkg = rospkg.rospack.RosPack().get_path('openpose_ros')
  ckpt_file = os.path.join(pkg, '_data', 'pose_iter_584000.ckpt')
 
  net_fn = functools.partial(pose_net_body_25,
                             part_stage=l2_stage-1,
                             limb_stage=l1_stage-1)
  pose_detector.initialize(net_fn, ckpt_file,
                           stage_n_L2, POSE_BODY_25_L2,
                           stage_n_L1, POSE_BODY_25_L1,
                           input_shape=input_shape,
                           allow_growth=True)
  
  for name in ['stage0', 'part_affinity_fields']:
    sparse = dense_to_sparse(pose_detector._end_points[name][0])
    pose_detector._end_points[name+'_sparse'] = sparse
    
  return EmptyResponse()

if __name__ == '__main__':
  import functools
  from nets import pose_net_body_25
  from labels import POSE_BODY_25_L1, POSE_BODY_25_L2
  
  bridge = CvBridge()
  rospy.init_node('openpose_tx2')

  pose_detector = KeyPointDetector()
  initialize_network()
    
  # (Neck, MidHip), (Neck, RShoulder), (Neck, LShoulder)
  #limbs_inds = np.array([0, 7, 11])
  # above and (RShoulder, RElbow), (LShoulder, LElbow)
  limbs_inds = np.array([0, 7, 11, 8, 12])
  limbs = np.array(pose_detector._limbs)[limbs_inds]
  affinity_inds = np.stack([limbs_inds*2,limbs_inds*2+1],1).flatten()

  image_sub = rospy.Subscriber('image', Image, callback,
                               queue_size=1, buff_size=1048576*8)
  mid_pub = rospy.Publisher('openpose_mid', SparseTensorArray, queue_size=1)
  people_pub = rospy.Publisher('people_tx2', PersonArray, queue_size=1)
  rospy.Service('initialize_network_tx2', Empty, initialize_network)

  time_pub = rospy.Publisher('processing_time_tx2', Duration, queue_size=1)

  pose_params = {}
  srv = ReconfServer(KeyPointDetectorConfig, reconf_callback)

  rospy.spin()
