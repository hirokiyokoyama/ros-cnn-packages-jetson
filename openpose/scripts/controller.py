#! /usr/bin/env python

import rospy
from openpose_ros.msg import PersonArray, Person
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
from dynamic_reconfigure.client import Client as ReconfClient
from std_msgs.msg import Duration
from std_srvs.srv import Empty
from geometry_msgs.msg import PoseWithCovarianceStamped
from camera_controller import CameraController

from _visualize import draw_peoplemsg

def draw_pwcsmsg(img, msg):
  p = msg.pose.pose.position
  try:
    u, v = cam.to_pixel([p.x, p.y, p.z], msg.header.frame_id)
  except:
    return
  u = u / cam._camera_model.width * img.shape[1]
  v = v / cam._camera_model.height * img.shape[0]
  center = (int(u), int(v))
  C = np.reshape(msg.pose.covariance, (6,6))[1:3,1:3]
  eigvals, eigvecs = np.linalg.eig(C)
  _x, _y = eigvecs[0,:]
  axes = np.sqrt(eigvals) * 20.
  angle = np.arctan2(_y, _x) / np.pi * 180
  #axes = (C[0,0] * 10, C[1,1] * 10)
  cv2.ellipse(img, (center, axes, angle), (128,64,128), -1)

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

def put_image(scrn, img, p):
  x, y = p
  h, w = img.shape[:2]
  scrn[y:(y+h), x:(x+w), :] = img

def put_stage(scrn, p, name):
  cv2.rectangle(scrn, p, (p[0]+100, p[1]+50), (255,255,255), -1)
  cv2.rectangle(scrn, p, (p[0]+100, p[1]+50), (0,0,0), 2)
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(scrn, name, (p[0]+10, p[1]+30),
              font, 0.5, (0,0,0), 1, cv2.LINE_AA)

def put_arrow(scrn, p1, p2, *p):
  color = (96,96,96)
  thickness = 2
  q1 = p1
  for q2 in (p2,)+p:
    cv2.line(scrn, q1, q2, color, thickness)
    q1 = q2
    
  q_2, q_1 = ((p1,p2)+p)[-2:]
  d = np.array(q_1) - np.array(q_2)
  d = d / np.linalg.norm(d)
  e = np.array([d[1], -d[0]])
  cv2.line(scrn, q_1, tuple(np.int32(q_1-10*d+5*e)), color, thickness)
  cv2.line(scrn, q_1, tuple(np.int32(q_1-10*d-5*e)), color, thickness)
  
def render(screen):
  H, W = screen.shape[:2]
  wi, hi = 240, 180
  wb, hb = 100, 50
  x0, y0 = 80, 50
  font = cv2.FONT_HERSHEY_SIMPLEX
  
  screen[:,:] = [96,176,224]

  tx2_stage = None
  tx2_end = None
  try:
    tx2_stage = rospy.get_param('/openpose_tx2/l1_stage')
  except:
    pass

  put_arrow(screen, (x0,y0+hi//2), (x0-40,y0+hi//2), (x0-40,H//2), (x0,H//2))
  x = x0
  y = H//2
  put_arrow(screen, (x+wb,y), (x+wb+50,y))
  put_stage(screen, (x,y-hb//2), 'CPM')
  x += wb+50
  for i in range(4):
    put_arrow(screen, (x+wb,y), (x+wb+50,y))
    put_stage(screen, (x,y-hb//2), 'Stage{}_L2'.format(i))
    x += wb+50
    if i == tx2_stage-1:
      tx2_end = x - 50
  put_arrow(screen, (x+wb,y), (x+wb+50,y))
  put_stage(screen, (x,y-hb//2), 'Stage0_L1')
  x += wb+50
  put_stage(screen, (x,y-hb//2), 'Stage1_L1')
  x += wb+50
  xavier_end = x - 50

  if image is not None:
    orig_image = cv2.resize(image, (wi, hi))
  else:
    orig_image = np.zeros((hi, wi, 3), dtype=np.uint8)
  put_image(screen, orig_image, (x0, y0))
  orig_w = rospy.get_param('openpose_image_width', 360)
  orig_h = rospy.get_param('openpose_image_height', 270)
  orig_size = orig_w * orig_h * 3
  cv2.putText(screen, 'CameraImage({}x{}): {} bytes'.format(orig_w, orig_h, orig_size),
              (x0, y0-10), font, 0.4, (0,0,0), 1, cv2.LINE_AA)

  if tx2_end:
    _x, _y = tx2_end-wb//2, H-50-hi
    put_arrow(screen, (_x,y+hb//2), (_x, _y))  
    #put_arrow(screen, (tx2_end-wb//2,y+hb//2), (tx2_end-wb//2, _y+hi//2), (tx2_end+30, _y+hi//2))
    tx2_image = orig_image.copy()
    if tx2_people is not None:
      draw_peoplemsg(tx2_image, tx2_people)
      tx2_image = tx2_image//2 + orig_image//2
    if tracked_person is not None:
      _tx2_image = tx2_image.copy()
      draw_pwcsmsg(_tx2_image, tracked_person)
      tx2_image = _tx2_image//2 + tx2_image//2
    put_image(screen, tx2_image, (_x-wi//2, _y))

    _x, _y = xavier_end-wb//2, H-50-hi
    put_arrow(screen, (_x,y+hb//2), (_x, _y))  
    xavier_image = orig_image.copy()
    if xavier_people is not None:
      draw_peoplemsg(xavier_image, xavier_people)
      xavier_image = xavier_image//2 + orig_image//2
    put_image(screen, xavier_image, (_x-wi//2, _y))
    
    cv2.rectangle(screen,
                  (x0-20, H//2-60),
                  (tx2_end+20, H//2+60),
                  (0,0,255), 2)
    cv2.rectangle(screen,
                  (tx2_end+30, H//2-60),
                  (xavier_end+30, H//2+60),
                  (255,0,0), 2)
    
  cv2.putText(screen, 'Edge device', (x0-20+10, H//2-45),
              font, 0.5, (0,0,255), 1, cv2.LINE_AA)
  cv2.putText(screen, 'Server', (tx2_end+30+10, H//2-45),
              font, 0.5, (255,0,0), 1, cv2.LINE_AA)

  if tx2_time is not None:
    ms = int(tx2_time.data.to_sec() * 1000)
    cv2.putText(screen, 'Processing time: {} ms'.format(ms),
                (x0-20+10, H//2+50),
                font, 0.4, (0,0,0), 1, cv2.LINE_AA)
  if xavier_time is not None:
    ms = int(xavier_time.data.to_sec() * 1000)
    cv2.putText(screen, 'Processing time: {} ms'.format(ms),
                (tx2_end+30+10, H//2+50),
                font, 0.4, (0,0,0), 1, cv2.LINE_AA)
  if sparse_tensor_sizes is not None:
    y = H//2-85
    for k, v in sparse_tensor_sizes.items():
      if k=='stage0':
        k = 'CPM'
      else:
        s = int(k[5])
        k = 'Stage{}_L2'.format(s-1)
      cv2.putText(screen, '{}: {} bytes'.format(k, v),
                  (tx2_end-30, y),
                  font, 0.4, (0,0,0), 1, cv2.LINE_AA)
      y += 12

if __name__ == '__main__':
  rospy.init_node('openpose_controller')
  cam = CameraController()
  move_camera = False
  
  bridge = CvBridge()
  image = None
  xavier_people = None
  tx2_people = None
  xavier_time = None
  tx2_time = None
  tracked_person = None
  sparse_tensor_sizes = None

  try:
    reconf_tx2 = ReconfClient('openpose_tx2', timeout=5.)
    reconf_xavier = ReconfClient('openpose_xavier', timeout=5.)
    reconf_tx2.update_configuration({
      'key_point_threshold': 0.3,
      'affinity_threshold': 0.1,
      'line_division': 3
    })
    reconf_xavier.update_configuration({
      'key_point_threshold': 0.2,
      'affinity_threshold': 0.1,
      'line_division': 5
    })
  except:
    rospy.logwarn('Failed to configure openpose nodes.')

  rospy.Subscriber('image', Image, img_cb,
                   queue_size=1, buff_size=1048576*8, tcp_nodelay=True)
  rospy.Subscriber('people_xavier', PersonArray, people_cb1,
                   queue_size=1, buff_size=1048576*8, tcp_nodelay=True)
  rospy.Subscriber('people_tx2', PersonArray, people_cb2,
                   queue_size=1, buff_size=1048576*8, tcp_nodelay=True)

  def _cb(msg):
    global tx2_time
    tx2_time = msg
  rospy.Subscriber('processing_time_tx2', Duration, _cb)
  def _cb(msg):
    global xavier_time
    xavier_time = msg
  rospy.Subscriber('processing_time_xavier', Duration, _cb)
  def _cb(msg):
    global tracked_person
    tracked_person = msg
  rospy.Subscriber('person', PoseWithCovarianceStamped, _cb)

  def _sizeof_sparse_tensor(msg):
    size = 14 #width to max_value
    size += len(msg.x_indices)
    size += len(msg.y_indices)
    size += len(msg.channel_indices)
    size += len(msg.quantized_values)
    return size
  def _cb(msg):
    global sparse_tensor_sizes
    sizes = {st.name: _sizeof_sparse_tensor(st) for st in msg.sparse_tensors}
    sparse_tensor_sizes = sizes
  from openpose_ros.msg import SparseTensorArray
  rospy.Subscriber('openpose_mid', SparseTensorArray, _cb,
                   queue_size=1, buff_size=1048576*4, tcp_nodelay=True)

  screen = np.zeros([700, 1200, 3], dtype=np.uint8)
  while not rospy.is_shutdown():
    render(screen)
    cv2.imshow('Openpose Demo', screen)
    
    key = cv2.waitKey(1)
    if key in range(ord('1'), ord('5')):
      stage = key - ord('0')
      rospy.set_param('/openpose_tx2/l1_stage', stage)
      rospy.ServiceProxy('initialize_network_tx2', Empty).call()
    if key == ord(' '):
      move_camera = not move_camera
      if move_camera:
        rospy.ServiceProxy('enable_camera_movement', Empty).call()
      else:
        rospy.ServiceProxy('disable_camera_movement', Empty).call()
    if key == ord('s'):
      cv2.imwrite('/data/screen.png', screen)
