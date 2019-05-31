import rospy
import numpy as np
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo
import tf
import tf2_ros
import tf2_geometry_msgs
from std_msgs.msg import Float64
import image_geometry

TILT_MIN = -0.3
TILT_MAX = 0.3
PAN_MIN = -2.5
PAN_MAX = 2.5

class CameraController:
    def __init__(self, tf_buffer=None):
        if tf_buffer is None:
            self._tf_buffer = tf2_ros.Buffer()
            tf2_ros.TransformListener(self._tf_buffer)
        else:
            self._tf_buffer = tf_buffer

        self._camera_model = image_geometry.PinholeCameraModel()
        camera_info = rospy.wait_for_message('camera_info', CameraInfo)
        self._camera_model.fromCameraInfo(camera_info)
        self._pan_pub = rospy.Publisher(
            '/pan_controller/command', Float64, queue_size=1)
        self._tilt_pub = rospy.Publisher(
            '/tilt_controller/command', Float64, queue_size=1)

    def look_at(self, xyz,
                source_frame='map',
                source_stamp=rospy.Time(0.)):
        if self._camera_model.projectionMatrix() is None:
            return None
        
        ps = PointStamped()
        ps.point.x, ps.point.y, ps.point.z = xyz
        ps.header.stamp = source_stamp
        ps.header.frame_id = source_frame
        p = self._tf_buffer.transform(
            ps, 'base_link', rospy.Duration(1.)).point
        rospy.loginfo('p={}'.format(p))
        pan = np.arctan2(p.y, p.x)
        tilt = np.arctan2(p.z-.1, p.x)
        rospy.loginfo('pan={}, tilt={}'.format(pan, tilt))
        self.move(pan=pan, tilt=tilt)

    def to_pixel(self, xyz, source_frame='map'):
        if self._camera_model.projectionMatrix() is None:
            return None

        ps = PointStamped()
        ps.point.x, ps.point.y, ps.point.z = xyz
        ps.header.stamp = rospy.Time(0.)
        ps.header.frame_id = source_frame
        p = self._tf_buffer.transform(ps, self._camera_model.tfFrame(),
                                      rospy.Duration(1.)).point
        return self._camera_model.project3dToPixel((p.x, p.y, p.z))

    def from_pixel(self, uv=None, target_frame='map'):
        if self._camera_model.projectionMatrix() is None:
            return None

        if uv is None:
            w = self._camera_model.width
            h = self._camera_model.height
            uv = (w/2, h/2)

        tr = self._tf_buffer.lookup_transform(target_frame,
                                              self._camera_model.tfFrame(),
                                              rospy.Time(0.),
                                              rospy.Duration(1.))
        q = tr.transform.rotation
        v = tr.transform.translation
        rot = tf.transformations.quaternion_matrix((q.x, q.y, q.z, q.w))
        xyz = self._camera_model.projectPixelTo3dRay(uv)
        n = np.dot(rot[:3,:3], xyz)
        return (v.x, v.y, v.z), (n[0], n[1], n[2])

    def move(self, pan=None, tilt=None):
        if tilt is not None and np.isnan(tilt):
            tilt = None
        if pan is not None and np.isnan(pan):
            pan = None
            
        if tilt is not None:
            tilt = min(TILT_MAX, max(TILT_MIN, tilt))
            self._tilt_pub.publish(Float64(tilt))

        if pan is not None:
            pan = pan % (np.pi*2)
            if pan > np.pi:
                pan -= np.pi*2
            pan = min(PAN_MAX, max(PAN_MIN, pan))
            self._pan_pub.publish(Float64(pan))

    def command(self, pan, tilt):
        self._pan_pub.publish(Float64(pan))
        self._tilt_pub.publish(Float64(tilt))
