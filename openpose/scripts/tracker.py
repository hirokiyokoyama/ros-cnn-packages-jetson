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

if __name__ == '__main__':
  rospy.init_node('openpose_tracker')

  cam = CameraController()
  track = None
  camera_target = None
  
  bridge = CvBridge()
  image = None
  xavier_people = None
  tx2_people = None

  reconf = ReconfClient('openpose_xavier')
  reconf.update_configuration({'affinity_threshold': 0.1})

  pose_pub = rospy.Publisher('person', PoseStamped, queue_size=1)
  pose2_pub = rospy.Publisher('camera_target', PoseStamped, queue_size=1)

  rospy.Subscriber('image', Image, img_cb,
                   queue_size=1, buff_size=1048576*8, tcp_nodelay=True)
  rospy.Subscriber('people_xavier', PersonArray, people_cb1,
                   queue_size=1, buff_size=1048576*8, tcp_nodelay=True)
  rospy.Subscriber('people_tx2', PersonArray, people_cb2,
                   queue_size=1, buff_size=1048576*8, tcp_nodelay=True)
  
  while not rospy.is_shutdown():
    if tx2_people is not None:
      detections = []
      for p in tx2_people.people:
        parts = {x.name: (x.x * 640, x.y * 480) for x in p.body_parts}
        if 'Neck' not in parts or 'MidHip' not in parts:
          continue
        _, neck = cam.from_pixel(parts['Neck'], 'base_link')
        _, hip = cam.from_pixel(parts['MidHip'], 'base_link')
        detections.append((neck, hip, len(p.body_parts), tx2_people.header.stamp))

      if detections:
        if track is None:
          track = max(detections, key=lambda d: d[2])
        else:
          track = max(detections, key=lambda d: np.dot(d[0], track[0]))

    p, _ = cam.from_pixel((0, 0), 'base_link')
    
    if track is not None:
      ps = PoseStamped()
      ps.header.stamp = track[3]
      ps.header.frame_id = 'base_link'
      ps.pose = ray_to_pose(p, track[0])
      pose_pub.publish(ps)

      if camera_target is None:
        camera_target = np.array(track[0])
      else:
        camera_target = camera_target*0.5 + np.array(track[0])*0.5

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
