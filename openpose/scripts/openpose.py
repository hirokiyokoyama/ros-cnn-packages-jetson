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

from sensor_msgs.msg import Image
from dynamic_reconfigure.server import Server

from openpose_ros.msg import Person, PersonArray, KeyPoint
from openpose_ros.srv import DetectKeyPoints, DetectKeyPointsResponse, DetectPeople, DetectPeopleResponse
from openpose_ros.msg import SparseTensor
from openpose_ros.srv import Compute, ComputeResponse
from openpose_ros.cfg import KeyPointDetectorConfig
from std_srvs.srv import Empty, EmptyResponse
from nets import non_maximum_suppression, connect_parts, pose_net_coco, hand_net, face_net
from labels import POSE_COCO_L1, POSE_COCO_L2, HAND
FACE = ['face_%02d' % (i+1) for i in range(71)]

stage = 6

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

  def initialize(self, stage=6):
    stage_n_L1 = 'stage%d_L1' % stage
    stage_n_L2 = 'stage%d_L2' % stage
    stage_n = 'stage%d' % stage
    
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
  msg.x_indices = inds[1]
  msg.y_indices = inds[0]
  msg.channel_indices = inds[2]
  val = tensor[inds]
  min_val = val.min()
  max_val = val.max()
  msg.min_value = min_val
  msg.max_value = max_val
  msg.quantized_values = np.uint8((val-min_val)/(max_val-min_val) * 255)
  return msg

def decode_sparse_tensor(msg):
  tensor = np.zeros(
    shape=[msg.height, msg.width, msg.channels],
    dtype=np.float32)
  val = msg.quantized_values/255. * (msg.max_value - msg.min_value) + msg.min_value
  tensor[(msg.y_indices, msg.x_indices, msg.channel_indices)] = val
  return tensor

def compute(req):
  # determine tensor to feed
  if req.input == 'image':
    input_stage = 0
    cv_image = bridge.imgmsg_to_cv2(req.image, 'rgb8')
    cv_image = cv_image/255.
    
    if pose_detector.input_shape is not None:
      cv_image = cv2.resize(cv_image, pose_detector.input_shape[::-1])
    image_batch = np.expand_dims(cv_image, 0)
    feed_dict = {pose_detector.ph_x: image_batch}
  else:
    if not req.input.startswith('stage'):
      raise ValueError('Argument "input" must be "image" or starts with "stage".')
    if req.input[5:] not in ['1', '2', '3', '4', '5', '6']:
      raise ValueError('Argument "input" specifies illegal stage.')
    input_stage = int(req.input[5:])
    feed_dict = {
      pose_detector.end_points[req.input+'_L1']: [decode_sparse_tensor(req.affinity_field)],
      pose_detector.end_points[req.input+'_L2']: [decode_sparse_tensor(req.confidence_map)],
      pose_detector.end_points['stage0']: [decode_sparse_tensor(req.feature_map)]
    }
  feed_dict[pose_detector.ph_threshold] = pose_params['key_point_threshold']
  
  # determine tensor to fetch
  if req.output == 'people':
    output_stage = 6
    fetch_list = [pose_detector.heat_map,
                  pose_detector.affinity,
                  pose_detector.keypoints]
    rospy.loginfo('Start processing.')
    heat_map, affinity, keypoints = pose_detector.sess.run(fetch_list, feed_dict)
    rospy.loginfo('Done.')
    # TODO: Scale is not always 8. It varies according to the preprocessing.
    scale_x = 8.
    scale_y = 8.
    inlier_lists = []
    for _,y,x,c in keypoints:
      x = x*scale_x + scale_x/2
      y = y*scale_y + scale_y/2
      inlier_lists.append((x,y))
      
    persons = connect_parts(affinity[0], keypoints[:,1:], pose_detector.limbs,
                            line_division = pose_params['line_division'],
                            threshold = pose_params['affinity_threshold'])
    persons = [{pose_detector.part_names[k]:inlier_lists[v] \
                for k,v in person.items()} for person in persons]
    msg = PersonArray()
    msg.people = [Person(body_parts=[KeyPoint(name=k, x=x, y=y) \
                                     for k,(x,y) in p.items()]) \
                  for p in persons]
    return ComputeResponse(people=msg)
  else:
    if not req.output.startswith('stage'):
      raise ValueError('Argument "output" must be "people" or starts with "stage".')
    if req.output[5:] not in ['1', '2', '3', '4', '5', '6']:
      raise ValueError('Argument "output" specifies illegal stage.')
    output_stage = int(req.output[5:])
    if output_stage < input_stage:
      raise ValueError('Output stage must be greater than input stage.')
    fetch_list = [
      pose_detector.end_points[req.output+'_L1'],
      pose_detector.end_points[req.output+'_L2'],
      pose_detector.end_points['stage0']
    ]
    affinity, heat_map, feat_map = pose_detector.sess.run(fetch_list, feed_dict)
    return ComputeResponse(feature_map=encode_sparse_tensor(feat_map[0]),
                           affinity_field=encode_sparse_tensor(affinity[0]),
                           confidence_map=encode_sparse_tensor(heat_map[0]))

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
  parser.add_argument('--enable-people', type=bool, default=True)
  parser.add_argument('--enable-hand', type=bool, default=True)
  parser.add_argument('--enable-face', type=bool, default=True)
  args, _ = parser.parse_known_args()

  pkg = rospkg.rospack.RosPack().get_path('openpose_ros')
  bridge = CvBridge()
  rospy.init_node('openpose')
  stage = rospy.get_param('~stage', 6)

  ckpt_file = os.path.join(pkg, 'data', 'pose_coco.ckpt')
  pose_detector = KeyPointDetector(pose_net_coco, ckpt_file, POSE_COCO_L2, POSE_COCO_L1, input_shape=(480, 640))
  if args.enable_people:
    pose_detector.initialize(stage=stage)
  pose_params = {}

  ckpt_file = os.path.join(pkg, 'data', 'face.ckpt')
  face_detector = KeyPointDetector(face_net, ckpt_file, FACE, input_shape=(480,640))
  if args.enable_face:
    face_detector.initialize(stage=stage)

  ckpt_file = os.path.join(pkg, 'data', 'hand.ckpt')
  hand_detector = KeyPointDetector(hand_net, ckpt_file, HAND, input_shape=(480,640))
  if args.enable_hand:
    hand_detector.initialize(stage=stage)

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

