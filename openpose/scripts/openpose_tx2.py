#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import rospy
import rospkg
import os
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from dynamic_reconfigure.server import Server as ReconfServer

from openpose_ros.msg import Person, PersonArray, KeyPoint
from openpose_ros.srv import DetectKeyPoints, DetectKeyPointsResponse, DetectPeople, DetectPeopleResponse
from openpose_ros.msg import SparseTensor, SparseTensorArray
from openpose_ros.srv import Compute, ComputeResponse
from openpose_ros.cfg import KeyPointDetectorConfig
from nets import non_maximum_suppression, connect_parts

class KeyPointDetector:
  def __init__(self):
    self._sess = None

  def initialize(self, net_fn, ckpt_file,
                 parts_tensor, part_names,
                 limbs_tensor=None, limbs=None,
                 input_shape=None):
    self.finalize()
    
    self._part_names = part_names
    if limbs is not None:
      self._limbs = [(part_names.index(p), part_names.index(q)) for p, q in limbs]
    else:
      self._limbs = None
    
    graph = tf.Graph()
    with graph.as_default():
      self._ph_x = tf.placeholder(tf.float32, shape=[None,None,None,3])
      
      if input_shape is None:
        x = self._ph_x
      else:
        x = tf.image.resize_images(self._ph_x, input_shape)
      self._end_points = net_fn(x)
      self._end_points['image'] = self._ph_x
      
      if parts_tensor in self._end_points:
        self._heat_map = self._end_points[parts_tensor]
      else:
        self._heat_map = graph.get_tensor_by_name(parts_tensor)
      self._end_points['confidence_maps'] = self._heat_map
        
      if limbs_tensor is None:
        self._affinity = tf.no_op()
      elif limbs_tensor in self._end_points:
        self._affinity = self._end_points[limbs_tensor]
      else:
        self._affinity = graph.get_tensor_by_name(limbs_tensor)
      self._end_points['part_affinity_fields'] = self._affinity
        
      self._ph_threshold = tf.placeholder(tf.float32)
      self._end_points['key_point_threshold'] = self._ph_threshold
      self._keypoints = non_maximum_suppression(
        self._heat_map[:,:,:,:-1], threshold=self._ph_threshold)
      self._end_points['key_points'] = self._keypoints
      
      saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=config)
    saver.restore(sess, ckpt_file)
    rospy.loginfo('Network was restored from {}.'.format(ckpt_file))
    self._sess = sess

  def finalize(self):
    if self._sess:
      sess = self._sess
      self._sess = None
      sess.close()

  def compute(self, fetch_list, feed_dict):
    if self._sess is None:
      raise ValueError('Not initialized.')
    
    remap = lambda x: self._end_points[x] if x in self._end_points else x
    fetch_list = list(map(remap, fetch_list))
    feed_dict = {remap(k):v for k,v in feed_dict.items()}
    return self._sess.run(fetch_list, feed_dict)

  def detect_keypoints(self, image,
                       key_point_threshold=0.5,
                       affinity_threshold=0.2,
                       line_division=15):
    if self._sess is None:
      raise ValueError('Not initialized.')
    
    orig_shape = image.shape
    image_batch = np.expand_dims(image/255., 0)

    predictions = [None, None, []]
    fetch = [self._heat_map, self._affinity, self._keypoints]
    rospy.loginfo('Start processing.')
    predictions = self._sess.run(fetch, {self._ph_x: image_batch,
                                         self._ph_threshold: key_point_threshold})
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
      persons = connect_parts(affinity[0], keypoints[:,1:], self._limbs,
                              line_division=line_division, threshold=affinity_threshold)
      persons = [{self._part_names[k]:inlier_lists[v] \
                  for k,v in person.items()} for person in persons]
    else:
      persons = [{self._part_names[c]:inliers \
                  for (_,_,_,c), inliers in zip(keypoints, inlier_lists)}]
    return persons

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
  
  fetch_list = ['stage0', 'part_affinity_fields']
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
    stage0 = encode_sparse_tensor(outputs[0][0], threshold=0.1)
    stage0.name = 'stage0'
    l1 = encode_sparse_tensor(outputs[1][0], threshold=0.1)
    l1.name = stage_n_L1
    msg.sparse_tensors = [stage0, l1]
    mid_pub.publish(msg)

  if publish_people:
    affinity = outputs[1][0]
    keypoints = outputs[2][:,1:]
    people = connect_parts(affinity, keypoints, pose_detector._limbs,
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
  if not msg.y_indices and msg.height*msg.width*msg.channels > 0:
    return val.reshape(msg.height, msg.width, msg.channels)
  tensor = np.zeros(
    shape=[msg.height, msg.width, msg.channels],
    dtype=np.float32)
  tensor[(np.fromstring(msg.y_indices, dtype=np.uint8),
          np.fromstring(msg.x_indices, dtype=np.uint8),
          np.fromstring(msg.channel_indices, dtype=np.uint8))] = val
  return tensor

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

if __name__ == '__main__':
  import functools
  from nets import pose_net_body_25
  from labels import POSE_BODY_25_L1, POSE_BODY_25_L2
  
  pkg = rospkg.rospack.RosPack().get_path('openpose_ros')
  bridge = CvBridge()
  rospy.init_node('openpose_tx2')
  l1_stage = rospy.get_param('~l1_stage', 4)
  l2_stage = rospy.get_param('~l2_stage', 2)
  stage_n_L1 = 'stage%d_L1' % l1_stage
  stage_n_L2 = 'stage%d_L2' % l2_stage

  ckpt_file = os.path.join(pkg, '_data', 'pose_iter_584000.ckpt')
  pose_detector = KeyPointDetector()
  net_fn = functools.partial(pose_net_body_25, part_stage=l2_stage-1, limb_stage=l1_stage-1)
  pose_detector.initialize(net_fn, ckpt_file,
                           stage_n_L2, POSE_BODY_25_L2,
                           stage_n_L1, POSE_BODY_25_L1,
                           input_shape=(300,400))
  pose_params = {}

  image_sub = rospy.Subscriber('image', Image, callback,
                               queue_size=1, buff_size=1048576*8)
  mid_pub = rospy.Publisher('openpose_mid', SparseTensorArray, queue_size=1)
  people_pub = rospy.Publisher('people_tx2', PersonArray, queue_size=1)

  srv = ReconfServer(KeyPointDetectorConfig, reconf_callback)
  rospy.spin()
