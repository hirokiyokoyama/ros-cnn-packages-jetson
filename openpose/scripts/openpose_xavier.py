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

from labels import POSE_BODY_25_L1, POSE_BODY_25_L2
limbs = [(POSE_BODY_25_L2.index(p), POSE_BODY_25_L2.index(q)) \
         for p, q in POSE_BODY_25_L1]

def sizeof_sparse_tensor(msg):
  size = 14 #width to max_value
  size += len(msg.x_indices)
  size += len(msg.y_indices)
  size += len(msg.channel_indices)
  size += len(msg.quantized_values)
  return size

def callback(data):
  publish_people = people_pub.get_num_connections() > 0
  if publish_people:
    fetch_list = ['part_affinity_fields', 'key_points']
    feed_dict = {'key_point_threshold': pose_params['key_point_threshold']}
    for st in data.sparse_tensors:
      feed_dict[st.name] = [decode_sparse_tensor(st)]
    t0 = rospy.Time.now()
    outputs = pose_detector.compute(fetch_list, feed_dict)
    t1 = rospy.Time.now()
    rospy.loginfo('Mid-stage to stage4: {} sec'.format((t1-t0).to_sec()))
    #people = extract_people(heat_map[0], affinity[0])
    
    affinity = outputs[0][0]
    keypoints = outputs[1][:,1:]
    people = connect_parts(affinity, keypoints, limbs,
                           line_division=pose_params['line_division'],
                           threshold=pose_params['affinity_threshold'])
    h, w = affinity.shape[:2]
    people = [ { pose_detector._part_names[k]: ((keypoints[v][1]+0.5)/w, (keypoints[v][0]+0.5)/h) \
                 for k,v in person.items() } for person in people ]
    t2 = rospy.Time.now()

    #TODO
    sizes = list(map(sizeof_sparse_tensor, data.sparse_tensors))
    print("Sizes: {} bytes\nTotal: {} bytes".format(sizes, sum(sizes)))
    rospy.loginfo("Post processing: {} sec".format((t2-t1).to_sec()))
    rospy.loginfo("Total: {} sec".format((t2-t0).to_sec()))
    print("Found {} people.".format(len(people)))

    msg = PersonArray()
    msg.header = data.header
    msg.people = [Person(body_parts=[KeyPoint(name=k, x=x, y=y) \
                                     for k,(x,y) in p.items()]) \
                  for p in people]
    people_pub.publish(msg)

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
  rospy.init_node('openpose_xavier')
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
                           input_shape=(300,400),
                           allow_growth=False)
  pose_params = {}

  st_sub = rospy.Subscriber('openpose_mid', SparseTensorArray, callback,
                            queue_size=1, buff_size=1048576*4, tcp_nodelay=True)
  people_pub = rospy.Publisher('people_xavier', PersonArray, queue_size=1)
  rospy.Service('compute_openpose', Compute, compute)

  srv = ReconfServer(KeyPointDetectorConfig, reconf_callback)
  rospy.spin()
