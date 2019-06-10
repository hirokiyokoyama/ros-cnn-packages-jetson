#! /usr/bin/env python

import rospy
from openpose_ros.msg import PersonArray, Person
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped
import tf
import numpy as np
from std_srvs.srv import Empty, EmptyResponse
from camera_controller import CameraController
from pykalman import KalmanFilter

FIXED_X = 2.

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

class Track:
  _kf = KalmanFilter()
  
  def __init__(self, t, x):
    self._mean = np.array(list(x)+[0.]*2)
    self._cov = np.diag([1.]*2+[2.]*2)
    self._t = t
    self._obs_mat = np.eye(2,4)
    self._obs_cov = np.eye(2) * .2**2

  def _transition_matrices(self, dt):
    trans_mat = np.array([[1., 0., dt, 0.],
                          [0., 1., 0., dt],
                          [0., 0., 1., 0.],
                          [0., 0., 0., 1.]])
    trans_cov = np.diag([.5**2 * dt]*2+[4.**2 * dt]*2)
    return trans_mat, trans_cov

  def predict(self, t):
    self._mean, self._cov = self.get_prediction(t)
    self._t = t

  def get_prediction(self, t):
    dt = (t - self._t).to_sec()
    t_mat, t_cov = self._transition_matrices(dt)
    mean, cov = Track._kf.filter_update(self._mean, self._cov,
                                        transition_matrix=t_mat,
                                        transition_covariance=t_cov)
    return mean, cov
    
  def update(self, t, obs, obs_cov=None):
    dt = (t - self._t).to_sec()
    t_mat, t_cov = self._transition_matrices(dt)
    if obs_cov is None:
      obs_cov = self._obs_cov
    mean, cov = Track._kf.filter_update(self._mean, self._cov, obs,
                                        transition_matrix=t_mat,
                                        transition_covariance=t_cov,
                                        observation_matrix=self._obs_mat,
                                        observation_covariance=obs_cov)
    self._mean, self._cov = mean, cov
    self._t = t

  def to_msg(self, t):
    msg = PoseWithCovarianceStamped()
    mean, cov = self.get_prediction(t)
    msg.header.stamp = t
    msg.header.frame_id = 'base_link'
    msg.pose.pose.position.x = FIXED_X
    msg.pose.pose.position.y = mean[0]
    msg.pose.pose.position.z = mean[1]
    msg.pose.pose.orientation.w = 1.
    msg.pose.covariance = [.1,        0,        0, 0, 0, 0,
                            0, cov[0,0], cov[0,1], 0, 0, 0,
                            0, cov[1,0], cov[1,1], 0, 0, 0,
                            0,        0,        0, 1, 0, 0,
                            0,        0,        0, 0, 1, 0,
                            0,        0,        0, 0, 0, 1]
    return msg

class Tracker:
  def __init__(self):
    self._tracks = []
    
  def update(self, t, detections):
    if detections:
      if not self._tracks:
        detection = max(detections, key=lambda d: d[0][1])
        self._tracks.append(Track(t, detection[0]))
      else:
        track = self._tracks[0]
        mean, cov = track.get_prediction(t)
        detection = min(detections, key=lambda d: np.linalg.norm(d[0]-mean[:2]))
        v = detection[0] - mean[:2]
        v = np.outer(v, v)
        track.update(t, detection[0], detection[1] + v * 10)

    tracks = []
    for track in self._tracks:
      _, cov = track.get_prediction(t)
      if np.trace(cov[:2,:2])/2. < 6.**2:
        tracks.append(track)
    self._tracks = tracks
    
if __name__ == '__main__':
  rospy.init_node('openpose_tracker')

  cam = CameraController()
  tracker = Tracker()
  camera_target = None
  move_camera = False
  people_msg = None
  
  pose_pub = rospy.Publisher('person', PoseWithCovarianceStamped, queue_size=1)
  pose2_pub = rospy.Publisher('camera_target', PoseStamped, queue_size=1)

  def _cb(msg):
    global people_msg
    people_msg = msg
  rospy.Subscriber('people_tx2', PersonArray, _cb,
                   queue_size=1, buff_size=1048576*8, tcp_nodelay=True)

  def _move(req):
    global move_camera
    move_camera = True
    return EmptyResponse()
  rospy.Service('enable_camera_movement', Empty, _move)
    
  def _stop(req):
    global move_camera
    move_camera = False
    return EmptyResponse()
  rospy.Service('disable_camera_movement', Empty, _stop)

  rate = rospy.Rate(100)
  prev_t = rospy.Time.now()
  while not rospy.is_shutdown():
    detections = []
    hand = None
    if people_msg is not None and people_msg.header.stamp > prev_t:
      for p in people_msg.people:
        parts = {x.name: np.array([x.x * 640, x.y * 480]) \
                 for x in p.body_parts}
        dist = lambda p, q: np.linalg.norm(p-q)
        if 'Neck' in parts and 'MidHip' in parts \
                and dist(parts['Neck'], parts['MidHip']) > 80:
          center = (parts['Neck'] + parts['MidHip']) / 2.
          cov = np.diag([0.5**2, 1.**2])
        elif 'RShoulder' in parts and 'RElbow' in parts \
                and dist(parts['RShoulder'], parts['RElbow']) > 40:
          center = parts['RShoulder']
          cov = np.diag([1.**2, 0.5**2])
        elif 'LShoulder' in parts and 'LElbow' in parts \
                and dist(parts['LShoulder'], parts['LElbow']) > 40:
          center = parts['LShoulder']
          cov = np.diag([1.**2, 0.5**2])
        else:
          continue
        try:
          _, center = cam.from_pixel(center, 'base_link')
        except:
          continue
        center = center / center[0] * FIXED_X       # assume x=FIXED_X and
        detections.append((center[1:], cov, parts)) # track on y-z plane

        if 'RShoulder' in parts and 'RElbow' in parts:
          shoulder = parts['RShoulder']
          elbow = parts['RElbow']
          d = elbow - shoulder
          th = np.arctan2(d[1], d[0])
          if th < -np.pi/4:
              hand = detections[-1]
      prev_t = people_msg.header.stamp
    if hand:
      tracker._tracks = [Track(prev_t, hand[0])]
    elif detections:
      tracker.update(prev_t, detections)
    else:
      tracker.update(rospy.Time.now(), [])

    try:
      p, camera_now = cam.from_pixel((0, 0), 'base_link')
    except:
      continue
      
    if tracker._tracks:
      track = tracker._tracks[0]
      t = rospy.Time.now()
      #ps = PoseStamped()
      #ps.header.stamp = t
      #ps.header.frame_id = 'base_link'
      mean, cov = track.get_prediction(t)
      x = np.array([FIXED_X, mean[0], mean[1]])
      #ps.pose = ray_to_pose(p, x)
      #pose_pub.publish(ps)
      pose_pub.publish(track.to_msg(t))

      if camera_target is None:
        camera_target = x
      else:
        camera_target[1] = camera_target[1]*0.9 + x[1]*0.1
        # sometimes tilt controller gives wrong value, so smooth z value
        camera_target[2] = camera_target[2]*0.95 + x[2]*0.05

      camera_target_vel = np.array([max(-1.5, min(+1.5, mean[0])),
                                    max(-0.5, min(+0.5, mean[1]))])
    else:
      # if there is no target, camera moves to the initial pose
      if camera_target is not None:
        camera_target[1] = camera_target[1] * 0.8
        camera_target[2] = camera_target[2] * 0.8
        
      # and publish empty PoseWithCovarianceStamped message
      msg = PoseWithCovarianceStamped()
      msg.header.stamp = rospy.Time.now()
      msg.header.frame_id = 'base_link'
      msg.pose.pose.orientation.w = 1.
      pose_pub.publish(msg)

      camera_target_vel = np.array([0., 0.])

    if camera_target is not None:
      ps = PoseStamped()
      ps.header.stamp = rospy.Time.now()
      ps.header.frame_id = 'base_link'
      ps.pose = ray_to_pose(p, camera_target)
      pose2_pub.publish(ps)

      pan = (camera_target_vel[0] - camera_now[1]) * .5
      tilt = (camera_target_vel[1] - camera_now[2]) * .3
      if move_camera:
        # pan: left, tilt: up
        #cam.command(pan, tilt)
        pan_speed = abs(camera_target[1] - camera_now[1]) * 10.
        pan_speed = max(0.5, min(5., pan_speed))
        tilt_speed = abs(camera_target[2] - camera_now[2]) * 5.
        tilt_speed = max(0.5, min(2., tilt_speed))
        cam.set_pan_speed(pan_speed)
        cam.set_tilt_speed(tilt_speed)
        cam.look_at(np.array(p) + camera_target,
                    source_frame='base_link')
        
    rate.sleep()
