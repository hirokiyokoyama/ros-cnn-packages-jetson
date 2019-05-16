#! /usr/bin/env python

import rospy
from openpose_ros.msg import PersonArray, Person
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseStamped
import tf
import numpy as np
from dynamic_reconfigure.client import Client as ReconfClient
from labels import POSE_BODY_25_L2, POSE_BODY_25_L1

from camera_controller import CameraController
from _visualize import draw_peoplemsg

def ray_to_pose(p, v):
  pose = Pose()
  pose.position.x, pose.position.y, pose.position.z = p
  w = np.zeros(3)
  w[np.argmin(np.square(v))] = 1.
  w = np.cross(v, w)
  w /= np.linalg.norm(w)
  mat = np.zeros([4,4], dtype=np.float)
  mat[:3,0] = v
  mat[:3,1] = w
  mat[:3,2] = np.cross(v, w)
  mat[3,3] = 1.
  q = tf.transformations.quaternion_from_matrix(mat)
  pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
  return pose

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

class Track:
  def __init__(self, t, x):
    x = np.array(x)
    self._x = x
    self._v = np.zeros(x.shape)
    self._t = t

  def predict(self, t):
    self._x, self._v = self.get_prediction(t)
    self._t = t

  def get_prediction(self, t):
    dt = (t - self._t).to_sec()
    x = self._x + self._v * dt
    return x, self._v
    
  def update(self, t, detection):
    dt = (t - self._t).to_sec()
    x = np.array(detection[1])
    v = (x - self._x) / dt
    _x, _v = self.get_prediction(t)
    self._x = _x * 0.2 + x * 0.8
    self._v = _v * 0.1 + v * 0.9
    self._t = t

class Tracker:
  def __init__(self):
    self._tracks = []
        
  def predict(self, t):
    for t in self._tracks:
      t.predict(t)
    
  def update(self, t, detections):
    if not detections:
      for track in self._tracks:
        track.predict(t)
      return
    if not self._tracks:
      detection = max(detections, key=lambda d: d[0])
      self._tracks.append(Track(t, detection[1]))
    else:
      track = self._tracks[0]
      detection = max(detections, key=lambda d: np.dot(d[2], track._x))
      track.update(t, detection)

if __name__ == '__main__':
  rospy.init_node('openpose_tracker')

  cam = CameraController()
  tracker = Tracker()
  camera_target = None
  
  bridge = CvBridge()
  image = None
  xavier_people = None
  tx2_people = None

  reconf_tx2 = ReconfClient('openpose_tx2')
  reconf_tx2.update_configuration({
      'key_point_threshold': 0.3,
      'affinity_threshold': 0.1,
      'line_division': 3
  })
  reconf_xavier = ReconfClient('openpose_xavier')
  reconf_xavier.update_configuration({
      'key_point_threshold': 0.2,
      'affinity_threshold': 0.1,
      'line_division': 5
  })

  pose_pub = rospy.Publisher('person', PoseStamped, queue_size=1)
  pose2_pub = rospy.Publisher('camera_target', PoseStamped, queue_size=1)

  rospy.Subscriber('image', Image, img_cb,
                   queue_size=1, buff_size=1048576*8, tcp_nodelay=True)
  rospy.Subscriber('people_xavier', PersonArray, people_cb1,
                   queue_size=1, buff_size=1048576*8, tcp_nodelay=True)
  rospy.Subscriber('people_tx2', PersonArray, people_cb2,
                   queue_size=1, buff_size=1048576*8, tcp_nodelay=True)

  rate = rospy.Rate(10)
  prev_t = rospy.Time.now()
  while not rospy.is_shutdown():
    detections = []
    t = rospy.Time.now()
    if tx2_people is not None and tx2_people.header.stamp > prev_t:
      for p in tx2_people.people:
        parts = {x.name: (x.x * 640, x.y * 480) for x in p.body_parts}
        if 'Neck' not in parts or 'MidHip' not in parts:
          continue
        _, neck = cam.from_pixel(parts['Neck'], 'base_link')
        _, hip = cam.from_pixel(parts['MidHip'], 'base_link')
        detections.append((len(p.body_parts), neck, hip))
      prev_t = tx2_people.header.stamp
      #t = prev_t
    tracker.update(t, detections)

    p, _ = cam.from_pixel((0, 0), 'base_link')
    
    if tracker._tracks:
      track = tracker._tracks[0]
      t = rospy.Time.now()
      ps = PoseStamped()
      ps.header.stamp = t
      ps.header.frame_id = 'base_link'
      x, v = track.get_prediction(t)
      ps.pose = ray_to_pose(p, x)
      pose_pub.publish(ps)

      if camera_target is None:
        camera_target = x
      else:
        camera_target = camera_target*0.5 + x*0.5

    if camera_target is not None:
      ps = PoseStamped()
      ps.header.stamp = rospy.Time.now()
      ps.header.frame_id = 'base_link'
      ps.pose = ray_to_pose(p, camera_target)
      pose2_pub.publish(ps)

      #cam.look_at(np.array(p) + camera_target,
      #            source_frame='base_link')
    
    if image is not None:
      xavier_image = image.copy()
      tx2_image = image.copy()
      if xavier_people is not None:
        draw_peoplemsg(xavier_image, xavier_people)
      if tx2_people is not None:
        draw_peoplemsg(tx2_image, tx2_people)
      cv2.imshow('Xavier', np.uint8(xavier_image*0.5+image*0.5))
      cv2.imshow('TX2', np.uint8(tx2_image*0.5+image*0.5))
    cv2.waitKey(1)
    rate.sleep()
