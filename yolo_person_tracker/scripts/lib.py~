import rospy
import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped as Pose
from geometry_msgs.msg import PointStamped, Point, Vector3Stamped, Vector3
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import ColorRGBA
import tf
import tf2_ros
import tf2_geometry_msgs
from nav_msgs.msg import OccupancyGrid
import threading
import actionlib
import image_geometry
from hsrb_interface import geometry

from saveable_object import SaveableObject

def rospose_to_tmcpose(pose):
    position = geometry.Vector3(x = pose.position.x,
                                y = pose.position.y,
                                z = pose.position.z)
    orientation = geometry.Quaternion(x = pose.orientation.x,
                                      y = pose.orientation.y,
                                      z = pose.orientation.z,
                                      w = pose.orientation.w)
    return position, orientation

def publish_initial_pose(pose, std=((0.5, 0.5), 15./180*np.pi),
                         topic='/laser_2d_correct_pose', ref_frame='map'):
    pub = rospy.Publisher(topic, Pose, queue_size=1, latch=True)
    msg = Pose()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = ref_frame
    msg.pose.pose.position.x = pose[0][0]
    msg.pose.pose.position.y = pose[0][1]
    msg.pose.pose.orientation.z = np.sin(pose[1]/2)
    msg.pose.pose.orientation.w = np.cos(pose[1]/2)
    msg.pose.covariance[0] = std[0][0]**2
    msg.pose.covariance[7] = std[0][1]**2
    msg.pose.covariance[35] = std[1]**2
    pub.publish(msg)
    #pub.unregister()

class LEDBlinker:
    def __init__(self, hz=100):
        self.color = ColorRGBA()
        self.count = 0
        self.hz = hz
        self.running = False

    def cb(self, msg):
        if msg.a == -1:
            return
        self.color = msg
        self.count = self.hz

    def run(self):
        rate = rospy.Rate(self.hz)
        pub = rospy.Publisher('/hsrb/command_status_led_rgb', ColorRGBA, queue_size=1)
        while not rospy.is_shutdown() and self.running:
            scale = float(self.count)/self.hz
            c = ColorRGBA()
            c.r = self.color.r * scale
            c.g = self.color.g * scale
            c.b = self.color.b * scale
            c.a = -1
            pub.publish(c)
            rate.sleep()
            if self.count > 0:
                self.count -= 1

    def __enter__(self):
        self.sub = rospy.Subscriber('/hsrb/command_status_led_rgb', ColorRGBA, self.cb)
        self.thread = threading.Thread(target=self.run)
        self.running = True
        self.thread.start()

    def __exit__(self, exctype, excval, traceback):
        self.sub.unregister()
        self.running = False
        self.thread = None

class OccupancyGridListener:
    def __init__(self, topic_name, tf_buffer=None):
        self.map_data = None
        if tf_buffer is None:
            self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1.))
            tf2_ros.TransformListener(self.tf_buffer)
        else:
            self.tf_buffer = tf_buffer
        rospy.Subscriber(topic_name, OccupancyGrid, self.map_cb)

    def map_cb(self, msg):
        if len(msg.data) != msg.info.width * msg.info.height:
            rospy.logerr('Data length does not match width*height.')
            return

        p = msg.info.origin.position
        q = msg.info.origin.orientation
        m = tf.transformations.quaternion_matrix((q.x, q.y, q.z, q.w))
        m[:3,3] = p.x, p.y, p.z
        s = 1./msg.info.resolution
        to_pixel = np.dot(np.array([[s,0,0,0],[0,s,0,0]]),
                          np.linalg.inv(m))
        data = np.reshape(msg.data, (msg.info.height, msg.info.width))

        self.map_data = (msg.header, to_pixel, data)

    def _to_affine(self, point, frame_id):
        if isinstance(point, PointStamped):
            p = self.tf_buffer.transform(point, frame_id,
                                         rospy.Duration(1.)).point
            affine = np.array([p.x, p.y, p.z, 1.])
        elif isinstance(point, Point):
            affine = np.array([point.x, point.y, point.z, 1.])
        else:
            affine = np.array([point[0], point[1], 0., 1.])
        return affine

    def get_occupancy(self, point):
        if self.map_data is None:
            return -1
        header, to_pixel, data = self.map_data
        affine = self._to_affine(point, header.frame_id)
        x, y = np.dot(to_pixel, affine)
        if x < 0 or x >= data.shape[1]\
           or y < 0 or y >= data.shape[0]:
            return -1
        return data[int(y), int(x)]

    def get_occupancy_on_line(self, p1, p2, n=None):
        if self.map_data is None:
            return -1
        header, to_pixel, data = self.map_data
        p1 = self._to_affine(p1, header.frame_id)
        p2 = self._to_affine(p2, header.frame_id)
        x1, y1 = np.dot(to_pixel, p1)
        x2, y2 = np.dot(to_pixel, p2)
        if n is None:
            d = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            n = max(2, int(np.ceil(d+1)))
        occupancy = np.zeros(n, dtype=np.int8)
        for i, s in enumerate(np.linspace(0, 1, n)):
            x = int((1-s) * x1 + s * x2)
            y = int((1-s) * y1 + s * y2)
            if x < 0 or x >= data.shape[1]\
               or y < 0 or y >= data.shape[0]:
                occupancy[i] =  -1
            else:
                occupancy[i] = data[y, x]
        return occupancy

#Joints: now 'arm_flex_joint' only
class HSRB_Impedance:
    def __init__(self, tf_buffer=None):
        from tmc_control_msgs.srv import ChangeExxxDriveMode

        self.arm_client = actionlib.SimpleActionClient('/hsrb/arm_trajectory_controller/follow_joint_trajectory',
                                                       FollowJointTrajectoryAction)
        self.arm_client.wait_for_server(rospy.Duration(1.))
        rospy.wait_for_service('/hsrb/drive_mode_controller/change_drive_mode')
        self.change_drive_mode = rospy.ServiceProxy('/hsrb/drive_mode_controller/change_drive_mode', ChangeExxxDriveMode)
        #self.config = 'compliance_soft'

    def soften_joints(self, joints):
        from tmc_control_msgs.msg import JointExxxDriveMode, ExxxDriveMode
        arg = JointExxxDriveMode()
        for joint in ['arm_flex_joint', 'wrist_flex_joint']:
            mode = ExxxDriveMode()
            mode.joint = joint
            mode.value = ExxxDriveMode.kJntPositionAndActVelocity
            if joint in joints:
                mode.value = ExxxDriveMode.kImpedance
            arg.drive_modes.append(mode)
        return self.change_drive_mode(arg).success

    def move_to_neutral(self, wait=False, timeout=None):
        #TODO: put neutral positions
        positions = {'arm_lift_joint':0,
                     'arm_flex_joint':0,
                     'arm_roll_joint':0,
                     'wrist_flex_joint':-np.pi/2,
                     'wrist_roll_joint':0}
        return self.move(wait=wait, timeout=timeout, **positions)

    def move_to_go(self, wait=False, timeout=None):
        #TODO: put to-go positions
        positions = {'arm_lift_joint':0,
                     'arm_flex_joint':0,
                     'arm_roll_joint':-np.pi/2,
                     'wrist_flex_joint':-np.pi/2,
                     'wrist_roll_joint':0}
        return self.move(wait=wait, timeout=timeout, **positions)

    def move(self, wait=False, timeout=None, **positions):
        ranges = {'arm_lift_joint':(0., 0.69),
                  'arm_flex_joint':(-2.617, 0.),
                  'arm_roll_joint':(-1.919, 3.665),
                  'wrist_flex_joint':(-1.919, 1.221),
                  'wrist_roll_joint':(-1.919, 3.665)}
        for k,(minimum, maximum) in ranges.items():
            if k in positions:
                positions[k] = min(maximum, max(minimum, positions[k]))

        arm_msg = rospy.wait_for_message('/hsrb/arm_trajectory_controller/state',
                                          JointTrajectoryControllerState, timeout)
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = positions.keys()
        p = JointTrajectoryPoint()
        p.positions = positions.values()
        for i, name in enumerate(arm_msg.joint_names):
            if name not in positions.keys():
                goal.trajectory.joint_names.append(name)
                p.positions.append(arm_msg.desired.positions[i])
        p.velocities = [0.]*len(p.positions)
        goal.trajectory.points = [p]
        p.time_from_start = rospy.Duration(1.)
        self.arm_client.send_goal(goal)

        if not wait:
            return True
        suc = actionlib.TerminalState.SUCCEEDED
        if self.arm_client.wait_for_result():
            return self.arm_client.get_state() == suc
        else:
            return False

class HSRB_Xtion:
    def __init__(self, tf_buffer=None):
        if tf_buffer is None:
            self.tf_buffer = tf2_ros.Buffer(rospy.Duration(15.))
            tf2_ros.TransformListener(self.tf_buffer)
        else:
            self.tf_buffer = tf_buffer

        self.head = actionlib.SimpleActionClient('/hsrb/head_trajectory_controller/follow_joint_trajectory',
                                                 FollowJointTrajectoryAction)
        self.head.wait_for_server(rospy.Duration(1.))
        self.arm = actionlib.SimpleActionClient('/hsrb/arm_trajectory_controller/follow_joint_trajectory',
                                                FollowJointTrajectoryAction)
        self.arm.wait_for_server(rospy.Duration(1.))
        self.camera_model = image_geometry.PinholeCameraModel()
        self.info_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/depth_registered/camera_info', CameraInfo, self.info_cb)

    def info_cb(self, msg):
        self.camera_model.fromCameraInfo(msg)

    def look_at(self, xyz,
                source_frame='map',
                source_stamp=rospy.Time(0.),
                wait=False, timeout=None, move_hand=True):
        if self.camera_model.projectionMatrix() is None:
            return None
        p = Point()
        p.x, p.y, p.z = xyz
        ps = PointStamped(point=p)
        ps.header.stamp = source_stamp
        ps.header.frame_id = source_frame
        p = self.tf_buffer.transform(ps, 'torso_lift_link',
                                     rospy.Duration(1.)).point
        rospy.loginfo('p={}'.format(p))
        pan = np.arctan2(p.y, p.x)
        tilt = np.arctan2(p.z-.15, p.x)
        rospy.loginfo('pan={}, tilt={}'.format(pan, tilt))
        return self.move(lift=None, pan=pan, tilt=tilt, wait=wait, timeout=timeout, move_hand=move_hand)

    def to_pixel(self, xyz, source_frame='map'):
        if self.camera_model.projectionMatrix() is None:
            return None

        p = Point()
        p.x, p.y, p.z = xyz
        ps = PointStamped(point=p)
        ps.header.stamp = rospy.Time(0.)
        ps.header.frame_id = source_frame
        p = self.tf_buffer.transform(ps, self.camera_model.tfFrame(),
                                     rospy.Duration(1.)).point
        return self.camera_model.project3dToPixel((p.x, p.y, p.z))

    def from_pixel(self, uv=None, target_frame='map'):
        if self.camera_model.projectionMatrix() is None:
            return None

        if uv is None:
            w = self.camera_model.width
            h = self.camera_model.height
            uv = (w/2, h/2)

        tr = self.tf_buffer.lookup_transform(target_frame,
                                             self.camera_model.tfFrame(),
                                             rospy.Time(0.),
                                             rospy.Duration(1.))
        q = tr.transform.rotation
        v = tr.transform.translation
        rot = tf.transformations.quaternion_matrix((q.x, q.y, q.z, q.w))
        xyz = self.camera_model.projectPixelTo3dRay(uv)
        n = np.dot(rot[:3,:3], xyz)
        return (v.x, v.y, v.z), (n[0], n[1], n[2])

    def move(self, wait=False, timeout=None, move_hand=True, lift=None, pan=None, tilt=None):
        if lift is not None and np.isnan(lift):
            lift = None
        if tilt is not None and np.isnan(tilt):
            tilt = None
        if pan is not None and np.isnan(pan):
            pan = None
        if lift is not None:
            lift = min(.69, max(0., lift))
        if tilt is not None:
            tilt = min(.52, max(-1.57, tilt))
        if pan is not None:
            pan = pan % (np.pi*2)
            if pan > 1.60: #1.74
                pan -= np.pi*2
            if pan < -3.59: #-3.83
                pan += np.pi*2
            if pan > 1.74:
                pan = 1.74
            #pan = min(1.74, max(-3.83, pan))
        arm_goal_sent = False
        head_goal_sent = False

        head_msg = None
        if tilt is not None or pan is not None:
            head_msg = rospy.wait_for_message('/hsrb/head_trajectory_controller/state',
                                              JointTrajectoryControllerState, timeout)
            goal = FollowJointTrajectoryGoal()
            goal.trajectory.joint_names = head_msg.joint_names
            p = head_msg.desired
            p.positions = list(p.positions)
            p.velocities = list(p.velocities)
            goal.trajectory.points = [p]
            if tilt is not None:
                i = head_msg.joint_names.index('head_tilt_joint')
                p.positions[i] = tilt
                p.velocities[i] = 0.
            if pan is not None:
                i = head_msg.joint_names.index('head_pan_joint')
                p.positions[i] = pan
                p.velocities[i] = 0.
            p.time_from_start = rospy.Duration(1.)
            self.head.send_goal(goal)
            head_goal_sent = True

        if lift is not None or True: #always needed because of arm_flex_joint
            msg = rospy.wait_for_message('/hsrb/arm_trajectory_controller/state',
                                         JointTrajectoryControllerState, timeout)
            goal = FollowJointTrajectoryGoal()
            goal.trajectory.joint_names = msg.joint_names
            p = msg.desired
            p.positions = list(p.positions)
            p.velocities = list(p.velocities)
            goal.trajectory.points = [p]
            if lift is not None:
                i = msg.joint_names.index('arm_lift_joint')
                p.positions[i] = lift
                p.velocities[i] = 0.
            else:
                i = msg.joint_names.index('arm_lift_joint')
                lift = p.positions[i]
            if move_hand and lift > 0.15:
                i = msg.joint_names.index('arm_flex_joint')
                j = msg.joint_names.index('arm_roll_joint')
                #p.positions[i] = (-2. - (-0.7)) * lift/.69 + (-0.7)
                if tilt is not None:
                    p.positions[i] = tilt - np.pi/2
                    p.positions[j] = -np.pi/2
                else:
                    if head_msg is None:
                        head_msg = rospy.wait_for_message('/hsrb/head_trajectory_controller/state',
                                                          JointTrajectoryControllerState, timeout)
                    j = head_msg.joint_names.index('head_tilt_joint')
                    p.positions[i] = head_msg.desired.positions[j] - np.pi/2
                    p.positions[j] = -np.pi/2
            if move_hand and p.positions[msg.joint_names.index('arm_lift_joint')] < 0.1:
                i = msg.joint_names.index('arm_flex_joint')
                j = msg.joint_names.index('arm_roll_joint')
                if p.positions[i] < -2.:
                    p.positions[i] = -2.
                    p.positions[j] = -np.pi/2

            p.time_from_start = rospy.Duration(1.)
            goal.trajectory.points = [p]
            self.arm.send_goal(goal)
            arm_goal_sent = True

        if not wait:
            arm_result = True
            head_result = True
        else:
            suc = actionlib.TerminalState.SUCCEEDED
            if not head_goal_sent:
                head_result = True
            else:
                if self.head.wait_for_result():
                    head_result = self.head.get_state() == suc
                else:
                    head_result = False
            if not arm_goal_sent:
                arm_result = True
            else:
                if self.arm.wait_for_result():
                    arm_result = self.arm.get_state() == suc
                else:
                    arm_result = False
        return arm_result, head_result

class PDFWriter:
    from reportlab.lib.pagesizes import A4
    def __init__(self, task_name='', trial = -1,
                 page_size=A4, team_name='Er@sers', time_str=None,
                 file_name=None):
        from reportlab.pdfgen import canvas
        from os.path import expanduser
        self.page_size = page_size
        if time_str is None:
            import time
            time_str = time.strftime("%d/%m/%y %H:%M:%S")
        if file_name is None:
            file_name = '~/'
            if task_name:
                file_name += task_name+'_'
            file_name += '{}'.format(team_name)
            if trial is not None:
                if trial < 0:
                    import os
                    files = os.listdir(expanduser(os.path.dirname(file_name)))
                    prefix = os.path.basename(file_name)
                    files = filter(lambda x: x.startswith(prefix), files)
                    max_trial = 0
                    for f in files:
                        f = f[len(prefix):]
                        if f.endswith('.pdf'):
                            f = f[:-4]
                        split = f.split('_')
                        if len(split) >=2 and split[1].isdigit():
                            max_trial = max(max_trial, int(split[1]))
                    trial = max_trial+1
                file_name += '_{}'.format(trial)
            #if len(time_str):
            #   file_name += '_'+time_str
            file_name += '.pdf'
        file_name = expanduser(file_name)
        self.canvas = canvas.Canvas(file_name, pagesize=page_size)
        self._y = page_size[1]
        rospy.on_shutdown(self.save)

        self.canvas.setFontSize(24)
        self.put_string('Team name: '+team_name)
        if trial is not None:
            self.put_string('Try number: {}'.format(trial))
        if len(time_str):
            self.put_string('Time: '+time_str)
        self.vspace(10)

    def put_string(self, string):
        if self._y < self.canvas._fontsize:
            self.next_page()
        self.canvas.drawString(0, self._y-self.canvas._fontsize, string)
        self._y -= self.canvas._fontsize * 1.1

    def put_image(self, rgb):
        import PIL.Image
        img = PIL.Image.fromarray(rgb)
        if self._y < img.size[1]:
            self.next_page()
        self.canvas.drawInlineImage(img, 0, self._y-img.size[1])
        self._y -= img.size[1]

    def vspace(self, space):
        self._y -= space

    def next_page(self):
        self.canvas.save()
        self._y = self.page_size[1]

    def save(self):
        self.next_page()

