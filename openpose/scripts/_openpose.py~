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
from openpose_ros.cfg import KeyPointDetectorConfig
from std_srvs.srv import Empty, EmptyResponse
from nets import non_maximum_suppression, connect_parts, pose_net_coco, hand_net
from labels import POSE_COCO_L1, POSE_COCO_L2, HAND

class KeyPointDetector:
  def __init__(self, net_fn, ckpt_file, part_names, limbs=None, input_shape=None, auto_detection_start=True):
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
    if auto_detection_start:
      self.initialize()

  def initialize(self):
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
        if 'stage6_L1' in self.end_points and 'stage6_L2' in self.end_points:
          self.affinity = self.end_points['stage6_L1']
          heat_map = self.end_points['stage6_L2']
        elif 'stage6' in self.end_points:
          self.affinity = tf.no_op()
          heat_map = self.end_points['stage6']
        else:
          raise ValueError('Network function must return a dictionary that contains "stage6" or both "stage6_L1" and "stage6_L2".')
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
    scale_x = 8.
    scale_y = 8.
                orig_shape = image.shape
    if self.input_shape is not None:
      #scale_x *= image.shape[1]/self.input_shape[1]
      #scale_y *= image.shape[0]/self.input_shape[0]
      image = cv2.resize(image, self.input_shape[::-1])
    image_batch = np.expand_dims(image, 0)

    predictions = [None, None, []]
    if self.sess:
      fetch = [self.heat_map, self.affinity, self.keypoints]
      rospy.loginfo('Start processing.')
      predictions = self.sess.run(fetch, {self.ph_x: image_batch, self.ph_threshold: key_point_threshold})
      #print('type(predictions):{}'.format(type(predictions)))
      rospy.loginfo('Done.')
    heat_map, affinity, keypoints = predictions
                
                scale_x = orig_shape[1]/float(heat_map.shape[2])
                scale_y = orig_shape[0]/float(heat_map.shape[1])
    #print('type(heat_map:{}, affinity:{}, keypoints:{})'.format(type(heat_map), type(affinity), type(keypoints)))
    #print('heat_map:: shape:{}, dtype:{})'.format(heat_map.shape, heat_map.dtype))
    #print('affinity:: shape:{}, dtype:{})'.format(affinity.shape, affinity.dtype))
    #print('keypoints:: shape:{}, dtype:{})'.format(keypoints.shape, keypoints.dtype))
    #markers = {}
    inlier_lists = []
    for _,y,x,c in keypoints:
      # if c not in markers:
      #   _, thresh = cv2.threshold(heat_map[0,:,:,c],
      #                 key_point_threshold, 255, 0)
      #   _, m = cv2.connectedComponents(np.uint8(thresh))
      #   markers[c] = m
      # else:
      #   m = markers[c]
      # ys, xs = np.where(m==m[y,x])
      # xs = xs*scale_x + scale_x/2
      # ys = ys*scale_y + scale_y/2
      # inlier_lists.append(np.vstack([xs,ys]).T)
      x = x*scale_x + scale_x/2
      y = y*scale_y + scale_y/2
      inlier_lists.append((x,y))

    if affinity is not None:
      persons = connect_parts(affinity[0], keypoints[:,1:], self.limbs,
                  line_division=line_division, threshold=affinity_threshold)
      # persons = [{self.part_names[k]:inlier_lists[v]\
      #       for k,v in person.iteritems()} for person in persons]
      persons = [{self.part_names[k]:inlier_lists[v]\
            for k,v in person.iteritems()} for person in persons]
    else:
      persons = [{self.part_names[c]:inliers\
            for (_,_,_,c), inliers in zip(keypoints, inlier_lists)}]
    return persons

  def visualize(self, image, persons):
    vis_image = image.copy()
    for i, person in enumerate(persons):
      hsv = (int(180.*i/len(persons)), 255, 255)
      bgr = tuple(cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0,0].tolist())
      for x,y in person.values():
        cv2.circle(vis_image, (int(x),int(y)), 5, bgr)
    cv2.imshow('OpenPose', vis_image)
    cv2.waitKey(1)

def callback(data):
  if not people_pub.get_num_connections() and not debug_mode:
    return
  try:
    cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')
  except CvBridgeError as e:
    rospy.logerr(e)
    return
  persons = pose_detector.detect_keypoints(cv_image, **pose_params)
  msg = PersonArray(header=data.header)
  msg.people = [Person(body_parts=[KeyPoint(name=k, x=x, y=y)\
                   for k,(x,y) in p.iteritems()])\
          for p in persons]
  people_pub.publish(msg)
  if show_prev:
    pose_detector.visualize(cv_image, persons)

def detect_people(req):
  try:
    cv_image = bridge.imgmsg_to_cv2(req.image, 'bgr8')
  except CvBridgeError as e:
    rospy.logerr(e)
    return None
  persons = pose_detector.detect_keypoints(cv_image, **pose_params)
  msg = DetectPeopleResponse()
  msg.people = [Person(body_parts=[KeyPoint(name=k, x=x, y=y)\
                   for k,(x,y) in p.iteritems()])\
          for p in persons]
  return msg

def detect_hand(req):
  try:
    cv_image = bridge.imgmsg_to_cv2(req.image, 'bgr8')
  except CvBridgeError as e:
    rospy.logerr(e)
    return None
  key_points = hand_detector.detect_keypoints(cv_image, **pose_params)[0]
  msg = DetectKeyPointsResponse()
  msg.key_points = [KeyPoint(name=k, x=x, y=y)\
            for k,(x,y) in key_points.iteritems()]
  if show_prev:
    pose_detector.visualize(cv_image, [key_points])
  return msg

def reconf_callback(config, level):
  for key in ['key_point_threshold', 'affinity_threshold', 'line_division']:
    pose_params[key] = config[key]
  return config

def enable_people_detector(req):
  pose_detector.initialize()
  return EmptyResponse()

def enable_hand_detector(req):
  hand_detector.initialize()
  return EmptyResponse()

def disable_people_detector(req):
  pose_detector.finalize()
  return EmptyResponse()

def disable_hand_detector(req):
  hand_detector.finalize()
  return EmptyResponse()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--auto_detection_start', type=bool, default=True, help='start detectino automatically.')
  args, _ = parser.parse_known_args()

  pkg = rospkg.rospack.RosPack().get_path('openpose_ros')
  bridge = CvBridge()
  rospy.init_node('openpose')
  show_prev = rospy.get_param('~show_prev', False)
  debug_mode = rospy.get_param('~debug_mode', False)
  rospy.loginfo('show_prev: {}, debug_mode: {}'.format(show_prev, debug_mode))

  ckpt_file = os.path.join(pkg, 'data', 'pose_coco.ckpt')
  pose_detector = KeyPointDetector(pose_net_coco, ckpt_file, POSE_COCO_L2, POSE_COCO_L1, input_shape=(256, 256), auto_detection_start=args.auto_detection_start)
  pose_params = {}

  #ckpt_file = os.path.join(pkg, 'data', 'face.ckpt')
  #face_detector = KeyPointDetector(face_net, ckpt_file, FACE)

  ckpt_file = os.path.join(pkg, 'data', 'hand.ckpt')
  hand_detector = KeyPointDetector(hand_net, ckpt_file, HAND, input_shape=(256,256), auto_detection_start=args.auto_detection_start)

  image_sub = rospy.Subscriber('image', Image, callback)
  people_pub = rospy.Publisher('people', PersonArray, queue_size=1)
  rospy.Service('detect_people', DetectPeople, detect_people)
  rospy.Service('detect_hand', DetectKeyPoints, detect_hand)
  #rospy.Service('detect_face', DetectKeyPoints, detect_face)
  rospy.Service('enable_people_detector', Empty, enable_people_detector)
  rospy.Service('enable_hand_detector', Empty, enable_hand_detector)
  rospy.Service('disable_people_detector', Empty, disable_people_detector)
  rospy.Service('disable_hand_detector', Empty, disable_hand_detector)

  srv = Server(KeyPointDetectorConfig, reconf_callback)
  #print('openpose start completed!')
  rospy.spin()

