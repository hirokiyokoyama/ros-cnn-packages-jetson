#!/usr/bin/env python

import rospy
import tf
import numpy as np
from yolo_ros.msg import ObjectArray
from yolo_ros.srv import DetectObjects
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseStamped
from yolo_ros.libs import TreeReader

from camera_controller import CameraController

CLASS = 'person'

def box_overlap(x1, w1, x2, w2):
    l1 = x1 - w1/2
    l2 = x2 - w2/2
    left = max(l1, l2)
    r1 = x1 + w1/2
    r2 = x2 + w2/2
    right = min(r1, r2)
    return right - left

def box_intersection(ax, ay, aw, ah, bx, by, bw, bh):
    w = box_overlap(ax, aw, bx, bw)
    h = box_overlap(ay, ah, by, bh)
    if w < 0 or h < 0:
        return 0
    return w*h

def box_union(ax, ay, aw, ah, bx, by, bw, bh):
    i = box_intersection(ax, ay, aw, ah, bx, by, bw, bh)
    u = aw*ah + bw*bh - i
    return u

def box_iou(ax, ay, aw, ah, bx, by, bw, bh):
    i = box_intersection(ax, ay, aw, ah, bx, by, bw, bh)
    u = aw*ah + bw*bh - i
    return float(i)/u

def obj_iou(obj1, obj2):
    return box_iou((obj1.left + obj1.right)/2,
                   (obj1.top + obj1.bottom)/2,
                   obj1.right - obj1.left,
                   obj1.bottom - obj1.top,
                   (obj2.left + obj2.right)/2,
                   (obj2.top + obj2.bottom)/2,
                   obj2.right - obj2.left,
                   obj2.bottom - obj2.top)

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

def callback(image):
    objects = detect_objects(image).objects
    objs = []
    for obj in objects.objects:
        if obj.objectness < 0.2:
            continue
        prob = tr.probability(obj.class_probability, CLASS)
        if prob < 0.2:
            continue
        _, v = cam.from_pixel(
            ((obj.left+obj.right)/2., (obj.top+obj.bottom)/2.),
            'base_link')

        if objs:
            iou = [obj_iou(obj, obj2) for obj2, _, _ in objs]
            i = np.argmax(iou)
            if iou[i] > .5:
                if objs[i][0].objectness < obj.objectness:
                    objs[i] = (obj, prob, v)
            else:
                objs.append((obj, prob, v))
        else:
            objs.append((obj, prob, v))

    global track, camera_target
    if objs:
        if track is None:
            track = max(objs, key=lambda o: o[1])
        else:
            track = max(objs, key=lambda o: np.dot(o[2], track[2]))

    p, _ = cam.from_pixel((0, 0), 'base_link')
    
    if track is not None:
        ps = PoseStamped()
        ps.header.stamp = image.header.stamp
        ps.header.frame_id = 'base_link'
        ps.pose = ray_to_pose(p, track[2])
        pose_pub.publish(ps)

        if camera_target is None:
            camera_target = np.array(track[2])
        else:
            camera_target = camera_target*0.5 + np.array(track[2])*0.5

    if camera_target is not None:
        ps = PoseStamped()
        ps.header.stamp = image.header.stamp
        ps.header.frame_id = 'base_link'
        ps.pose = ray_to_pose(p, camera_target)
        pose2_pub.publish(ps)

        cam.look_at(
            np.array(p) + camera_target,
            source_frame='base_link')
        
rospy.init_node('person_tracker')

bridge = CvBridge()
tr = TreeReader()
cam = CameraController()
track = None
camera_target = None
image_to_show = None

detect_objects = rospy.ServiceProxy('detect_objects', DetectObjects)
image_sub = rospy.Subscriber('image', Image, callback)
pose_pub = rospy.Publisher('detected_{}'.format(CLASS), PoseStamped, queue_size=1)
pose2_pub = rospy.Publisher('camera_target'.format(CLASS), PoseStamped, queue_size=1)

rospy.spin()
