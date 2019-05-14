#!/usr/bin/env python

import argparse
import cv2
import numpy as np
import tensorflow as tf
import rospy
import threading
import rospkg
import os
from cv_bridge import CvBridge, CvBridgeError
from distutils.util import strtobool

from sensor_msgs.msg import Image
from dynamic_reconfigure.server import Server

from openpose_ros.msg import Person, PersonArray, KeyPoint
from openpose_ros.srv import DetectKeyPoints, DetectKeyPointsResponse, DetectPeople, DetectPeopleResponse
from openpose_ros.msg import SparseTensor
from openpose_ros.srv import Compute, ComputeResponse
from openpose_ros.cfg import KeyPointDetectorConfig
from std_srvs.srv import Empty, EmptyResponse
from nets import non_maximum_suppression, connect_parts, pose_net_body_25, hand_net, face_net
from labels import POSE_BODY_25_L1, POSE_BODY_25_L2, HAND
FACE = ['face_%02d' % (i+1) for i in range(71)]

l2_stage = 1
l1_stage = 4

class KeyPointDetector:
  def __init__(self, net_fn, ckpt_file, part_names,
               limbs=None, input_shape=None):
    self.net_fn = net_fn
    self.ckpt_file = ckpt_file
    self.input_shape = input_shape
    self.sess = None
    self.graph = None
    self.part_names = part_names
    if limbs is not None:
      self.limbs = [(part_names.index(p), part_names.index(q)) for p, q in limbs]
    else:
      self.limbs = None

  def initialize(self, l2_stage=1, l1_stage=4):
    stage_n_L1 = 'stage%d_L1' % l1_stage
    stage_n_L2 = 'stage%d_L2' % l2_stage
    stage_n = 'stage%d' % l2_stage
    
    if self.sess:
      self.sess.close()
    if self.graph is None:
      self.graph = tf.Graph()
      with self.graph.as_default():
        shape = [None,None,None,3]
        if self.input_shape is not None:
          shape[1] = self.input_shape[0]
          shape[2] = self.input_shape[1]
        self.ph_x = tf.placeholder(tf.float32, shape=shape)
        self.end_points = self.net_fn(self.ph_x)
        self.end_points['image'] = self.ph_x
        if stage_n_L1 in self.end_points and stage_n_L2 in self.end_points:
          self.affinity = self.end_points[stage_n_L1]
          heat_map = self.end_points[stage_n_L2]
        elif stage_n in self.end_points:
          self.affinity = tf.no_op()
          heat_map = self.end_points[stage_n]
        else:
          raise ValueError('Network function must return a dictionary that contains "%s" or both "%s" and "%s".' % (stage_n, stage_n_L1, stage_n_L2))
        self.heat_map = heat_map
        self.ph_threshold = tf.placeholder(tf.float32)
        self.keypoints = non_maximum_suppression(heat_map[:,:,:,:-1], threshold=self.ph_threshold)
        saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(graph=self.graph, config=config)
    saver.restore(self.sess, self.ckpt_file)
    rospy.loginfo('Network was restored from {}.'.format(self.ckpt_file))

  def finalize(self):
    if self.sess:
      sess = self.sess
      self.sess = None
      sess.close()

  def detect_keypoints(self, image,
                       key_point_threshold=0.5,
                       affinity_threshold=0.2,
                       line_division=15):
    image = image/255.
    orig_shape = image.shape
    if self.input_shape is not None:
      image = cv2.resize(image, self.input_shape[::-1])
    image_batch = np.expand_dims(image, 0)

    predictions = [None, None, []]
    if not self.sess:
      return None
    fetch = [self.heat_map, self.affinity, self.keypoints]
    rospy.loginfo('Start processing.')
    predictions = self.sess.run(fetch, {self.ph_x: image_batch,
                                        self.ph_threshold: key_point_threshold})
    rospy.loginfo('Done.')
    heat_map, affinity, keypoints = predictions
    scale_x = orig_shape[1]/float(heat_map.shape[2])
    scale_y = orig_shape[0]/float(heat_map.shape[1])
    inlier_lists = []
    for _,y,x,c in keypoints:
      x = x*scale_x + scale_x/2
      y = y*scale_y + scale_y/2
      inlier_lists.append((x,y))

    if affinity is not None:
      persons = connect_parts(affinity[0], keypoints[:,1:], self.limbs,
                              line_division=line_division, threshold=affinity_threshold)
      persons = [{self.part_names[k]:inlier_lists[v] \
                  for k,v in person.items()} for person in persons]
    else:
      persons = [{self.part_names[c]:inliers \
                  for (_,_,_,c), inliers in zip(keypoints, inlier_lists)}]
    return persons

def callback(data):
  if not people_pub.get_num_connections():
    return
  try:
    cv_image = bridge.imgmsg_to_cv2(data, 'rgb8')
  except CvBridgeError as e:
    rospy.logerr(e)
    return
  persons = pose_detector.detect_keypoints(cv_image, **pose_params)
  msg = PersonArray(header=data.header)
  msg.people = [Person(body_parts=[KeyPoint(name=k, x=x, y=y) \
                                   for k,(x,y) in p.items()]) \
                for p in persons]
  people_pub.publish(msg)

def detect_people(req):
  try:
    cv_image = bridge.imgmsg_to_cv2(req.image, 'rgb8')
  except CvBridgeError as e:
    rospy.logerr(e)
    return None
  persons = pose_detector.detect_keypoints(cv_image, **pose_params)
  msg = DetectPeopleResponse()
  msg.people = [Person(body_parts=[KeyPoint(name=k, x=x, y=y) \
                                   for k,(x,y) in p.items()]) \
                for p in persons]
  return msg

def detect_keypoints(req, detector):
  try:
    cv_image = bridge.imgmsg_to_cv2(req.image, 'rgb8')
  except CvBridgeError as e:
    rospy.logerr(e)
    return None
  key_points = detector.detect_keypoints(cv_image, **pose_params)[0]
  msg = DetectKeyPointsResponse()
  msg.key_points = [KeyPoint(name=k, x=x, y=y) \
                    for k,(x,y) in key_points.items()]
  return msg

def detect_hand(req):
  return detect_keypoints(req, hand_detector)

def detect_face(req):
  return detect_keypoints(req, face_detector)

def encode_sparse_tensor(tensor, threshold=0.1, signed=True):
  msg = SparseTensor()
  msg.width = tensor.shape[1]
  msg.height = tensor.shape[0]
  msg.channels = tensor.shape[2]
  if signed:
    inds = np.where(np.abs(tensor) > threshold)
  else:
    inds = np.where(tensor > threshold)
  
  msg.x_indices = inds[1].tolist()
  msg.y_indices = inds[0].tolist()
  msg.channel_indices = inds[2].tolist()
  val = tensor[inds]
  if len(val) > 0:
    min_val = val.min()
    max_val = val.max()
    msg.min_value = min_val
    msg.max_value = max_val
    msg.quantized_values = np.uint8((val-min_val)/(max_val-min_val) * 255).tolist()
  return msg

def decode_sparse_tensor(msg):
  val = np.fromstring(msg.quantized_values, dtype=np.uint8)
  val = np.float32(val)/255. * (msg.max_value - msg.min_value) + msg.min_value
  if not msg.y_indices:
    return val.reshape(msg.height, msg.width, msg.channels)
  tensor = np.zeros(
    shape=[msg.height, msg.width, msg.channels],
    dtype=np.float32)
  tensor[(np.fromstring(msg.y_indices, dtype=np.uint8),
          np.fromstring(msg.x_indices, dtype=np.uint8),
          np.fromstring(msg.channel_indices, dtype=np.uint8))] = val
  return tensor

def compute(req):
  ep = pose_detector.end_points
  fetch_list = [ep[x] if x in ep else x for x in req.queries]
  feed_dict = {
    ep[x.name] if x.name in ep else x.name: [decode_sparse_tensor(x)] \
    for x in req.input_tensors }
  outputs = pose_detector.sess.run(fetch_list, feed_dict)
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

def enable_people_detector(req):
  pose_detector.initialize(stage=stage)
  return EmptyResponse()

def enable_face_detector(req):
  face_detector.initialize(stage=stage)
  return EmptyResponse()

def enable_hand_detector(req):
  hand_detector.initialize(stage=stage)
  return EmptyResponse()

def disable_people_detector(req):
  pose_detector.finalize()
  return EmptyResponse()

def disable_face_detector(req):
  face_detector.finalize()
  return EmptyResponse()

def disable_hand_detector(req):
  hand_detector.finalize()
  return EmptyResponse()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--enable-people', type=strtobool, default=True)
  parser.add_argument('--enable-hand', type=strtobool, default=True)
  parser.add_argument('--enable-face', type=strtobool, default=True)
  args, _ = parser.parse_known_args()

  pkg = rospkg.rospack.RosPack().get_path('openpose_ros')
  bridge = CvBridge()
  rospy.init_node('openpose')
  l1_stage = rospy.get_param('~l1_stage', 4)
  l2_stage = rospy.get_param('~l2_stage', 1)

  ckpt_file = os.path.join(pkg, '_data', 'pose_iter_584000.ckpt')
  pose_detector = KeyPointDetector(pose_net_body_25, ckpt_file, POSE_BODY_25_L2, POSE_BODY_25_L1, input_shape=(480,640))
  if args.enable_people:
    pose_detector.initialize(l2_stage=l2_stage, l1_stage=l1_stage)
  pose_params = {}

  ckpt_file = os.path.join(pkg, 'data', 'face.ckpt')
  face_detector = KeyPointDetector(face_net, ckpt_file, FACE, input_shape=(480,640))
  if args.enable_face:
    face_detector.initialize(l2_stage=l2_stage)

  ckpt_file = os.path.join(pkg, 'data', 'hand.ckpt')
  hand_detector = KeyPointDetector(hand_net, ckpt_file, HAND, input_shape=(480,640))
  if args.enable_hand:
    hand_detector.initialize(l2_stage=l2_stage)

  image_sub = rospy.Subscriber('image', Image, callback)
  people_pub = rospy.Publisher('people', PersonArray, queue_size=1)
  rospy.Service('detect_people', DetectPeople, detect_people)
  rospy.Service('detect_hand', DetectKeyPoints, detect_hand)
  rospy.Service('detect_face', DetectKeyPoints, detect_face)
  rospy.Service('enable_people_detector', Empty, enable_people_detector)
  rospy.Service('enable_hand_detector', Empty, enable_hand_detector)
  rospy.Service('disable_people_detector', Empty, disable_people_detector)
  rospy.Service('disable_hand_detector', Empty, disable_hand_detector)
  rospy.Service('compute_openpose', Compute, compute)

  srv = Server(KeyPointDetectorConfig, reconf_callback)
  rospy.spin()

