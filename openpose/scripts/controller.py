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

from _visualize import draw_peoplemsg

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
  
def render(screen):
  screen[:,:] = [96,176,224]

  if image is not None:
    orig_image = cv2.resize(image, (240, 180))
  else:
    orig_image = np.zeros((180, 240, 3), dtype=np.uint8)
  put_image(screen, orig_image, (50, 50))
  
  xavier_image = orig_image.copy()
  if xavier_people is not None:
    draw_peoplemsg(xavier_image, xavier_people)
    xavier_image = xavier_image//2 + orig_image//2
  put_image(screen, xavier_image, (50+240+50, 700-180-50))
        
  tx2_image = orig_image.copy()
  if tx2_people is not None:
    draw_peoplemsg(tx2_image, tx2_people)
    tx2_image = tx2_image//2 + orig_image//2
  put_image(screen, tx2_image, (1200-240-50, 700-180-50))

  tx2_stage = None
  tx2_end = None
  try:
    tx2_stage = rospy.get_param('/openpose_tx2/l1_stage')
  except:
    pass
  
  x = 50
  y = 700//2-25
  put_stage(screen, (x,y), 'CPM')
  x += 100+50
  for i in range(4):
    put_stage(screen, (x,y), 'Stage{}_L2'.format(i))
    x += 100+50
    if i == tx2_stage-1:
      tx2_end = x - 30
  put_stage(screen, (x,y), 'Stage0_L1')
  x += 100+50
  put_stage(screen, (x,y), 'Stage1_L1')
  x += 100+50
  xavier_end = x - 30

  if tx2_end:
    cv2.rectangle(screen,
                  (30, 700//2-60),
                  (tx2_end,700//2+60),
                  (0,0,255), 2)
    cv2.rectangle(screen,
                  (tx2_end+10, 700//2-60),
                  (xavier_end+10,700//2+60),
                  (255,0,0), 2)
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(screen, 'Edge device', (30+10, 700//2-45),
              font, 0.5, (0,0,0), 1, cv2.LINE_AA)
  cv2.putText(screen, 'Server', (tx2_end+10+10, 700//2-45),
              font, 0.5, (0,0,0), 1, cv2.LINE_AA)

  if tx2_time is not None:
    ms = int(tx2_time.data.to_sec() * 1000)
    cv2.putText(screen, 'Processing time: {} ms'.format(ms),
                (30+10, 700//2+50),
                font, 0.4, (0,0,0), 1, cv2.LINE_AA)
  if xavier_time is not None:
    ms = int(xavier_time.data.to_sec() * 1000)
    cv2.putText(screen, 'Processing time: {} ms'.format(ms),
                (tx2_end+10+10, 700//2+50),
                font, 0.4, (0,0,0), 1, cv2.LINE_AA)
  if sparse_tensor_sizes is not None:
    y = 700//2-80
    for k, v in sparse_tensor_sizes.items():
      if k=='stage0':
        k = 'CPM'
      else:
        s = int(k[5])
        k = 'Stage{}_L2'.format(s-1)
      cv2.putText(screen, '{}: {} bytes'.format(k, v),
                  (tx2_end-20, y),
                  font, 0.4, (0,0,0), 1, cv2.LINE_AA)
      y += 10

if __name__ == '__main__':
  rospy.init_node('openpose_controller')
  move_camera = False
  
  bridge = CvBridge()
  image = None
  xavier_people = None
  tx2_people = None
  xavier_time = None
  tx2_time = None
  sparse_tensor_sizes = None

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
