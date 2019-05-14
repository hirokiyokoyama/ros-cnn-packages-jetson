#!/usr/bin/env python
import rospy
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import scipy
from scipy import optimize

#(num_parts, num_limbs)=(19,19) for COCO, (num_parts, num_limbs)=(16,14) for MPI
def pose_net(x, num_parts=19, num_limbs=19, num_stages=6):
  end_points = {}

  with slim.arg_scope([slim.max_pool2d], stride=2, kernel_size=[2,2]):
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME',
              activation_fn=tf.nn.relu, normalizer_fn=None):
      net = slim.conv2d(x, 64, [3,3], scope='conv1_1')
      net = slim.conv2d(net, 64, [3,3], scope='conv1_2')
      net = slim.max_pool2d(net, scope='pool1_stage1')
      net = slim.conv2d(net, 128, [3,3], scope='conv2_1')
      net = slim.conv2d(net, 128, [3,3], scope='conv2_2')
      net = slim.max_pool2d(net, scope='pool2_stage1')
      net = slim.conv2d(net, 256, [3,3], scope='conv3_1')
      net = slim.conv2d(net, 256, [3,3], scope='conv3_2')
      net = slim.conv2d(net, 256, [3,3], scope='conv3_3')
      net = slim.conv2d(net, 256, [3,3], scope='conv3_4')
      net = slim.max_pool2d(net, scope='pool3_stage1')
      net = slim.conv2d(net, 512, [3,3], scope='conv4_1')
      net = slim.conv2d(net, 512, [3,3], scope='conv4_2')
      net = slim.conv2d(net, 256, [3,3], scope='conv4_3_CPM')
      net = slim.conv2d(net, 128, [3,3], scope='conv4_4_CPM')
      end_points['stage0'] = net

      limb = slim.conv2d(net, 128, [3,3], scope='conv5_1_CPM_L1')
      limb = slim.conv2d(limb, 128, [3,3], scope='conv5_2_CPM_L1')
      limb = slim.conv2d(limb, 128, [3,3], scope='conv5_3_CPM_L1')
      limb = slim.conv2d(limb, 512, [1,1], scope='conv5_4_CPM_L1')
      limb = slim.conv2d(limb, num_limbs*2, [1,1], activation_fn=None, scope='conv5_5_CPM_L1')
      end_points['stage1_L1'] = limb

      part = slim.conv2d(net, 128, [3,3], scope='conv5_1_CPM_L2')
      part = slim.conv2d(part, 128, [3,3], scope='conv5_2_CPM_L2')
      part = slim.conv2d(part, 128, [3,3], scope='conv5_3_CPM_L2')
      part = slim.conv2d(part, 512, [1,1], scope='conv5_4_CPM_L2')
      part = slim.conv2d(part, num_parts, [1,1], activation_fn=None, scope='conv5_5_CPM_L2')
      end_points['stage1_L2'] = part

  with slim.arg_scope([slim.conv2d], stride=1, padding='SAME',
            activation_fn=tf.nn.relu, normalizer_fn=None):
    for stage in range(2, num_stages+1):
      concat = tf.concat(values=[limb, part, net], axis=3, name='concat_stage{}'.format(stage))

      limb = slim.conv2d(concat, 128, [7,7], scope='Mconv1_stage{}_L1'.format(stage))
      limb = slim.conv2d(limb, 128, [7,7], scope='Mconv2_stage{}_L1'.format(stage))
      limb = slim.conv2d(limb, 128, [7,7], scope='Mconv3_stage{}_L1'.format(stage))
      limb = slim.conv2d(limb, 128, [7,7], scope='Mconv4_stage{}_L1'.format(stage))
      limb = slim.conv2d(limb, 128, [7,7], scope='Mconv5_stage{}_L1'.format(stage))
      limb = slim.conv2d(limb, 128, [1,1], scope='Mconv6_stage{}_L1'.format(stage))
      limb = slim.conv2d(limb, num_limbs*2, [1,1], activation_fn=None, scope='Mconv7_stage{}_L1'.format(stage))
      end_points['stage{}_L1'.format(stage)] = limb

      part = slim.conv2d(concat, 128, [7,7], scope='Mconv1_stage{}_L2'.format(stage))
      part = slim.conv2d(part, 128, [7,7], scope='Mconv2_stage{}_L2'.format(stage))
      part = slim.conv2d(part, 128, [7,7], scope='Mconv3_stage{}_L2'.format(stage))
      part = slim.conv2d(part, 128, [7,7], scope='Mconv4_stage{}_L2'.format(stage))
      part = slim.conv2d(part, 128, [7,7], scope='Mconv5_stage{}_L2'.format(stage))
      part = slim.conv2d(part, 128, [1,1], scope='Mconv6_stage{}_L2'.format(stage))
      part = slim.conv2d(part, num_parts, [1,1], activation_fn=None, scope='Mconv7_stage{}_L2'.format(stage))
      end_points['stage{}_L2'.format(stage)] = part

  #concat = tf.concat(values=[limb, part, net], axis=3, name='concat_stage{}'.format(num_stages+1))
  return end_points

def pose_net_coco(x):
  return pose_net(x, num_parts=19, num_limbs=19, num_stages=6)

def pose_net_mpi(x):
  return pose_net(x, num_parts=16, num_limbs=14, num_stages=6)

def pose_net_body_25(x, num_parts=26, num_limbs=26):
  def prelu(x):
    with tf.variable_scope(''):
      return tf.keras.layers.PReLU(shared_axes=[1,2])(x)
    
  end_points = {}

  with slim.arg_scope([slim.max_pool2d], stride=2, kernel_size=[2,2]):
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME',
              activation_fn=tf.nn.relu, normalizer_fn=None):
      net = slim.conv2d(x, 64, [3,3], scope='conv1_1')
      net = slim.conv2d(net, 64, [3,3], scope='conv1_2')
      net = slim.max_pool2d(net, scope='pool1_stage1')
      net = slim.conv2d(net, 128, [3,3], scope='conv2_1')
      net = slim.conv2d(net, 128, [3,3], scope='conv2_2')
      net = slim.max_pool2d(net, scope='pool2_stage1')
      net = slim.conv2d(net, 256, [3,3], scope='conv3_1')
      net = slim.conv2d(net, 256, [3,3], scope='conv3_2')
      net = slim.conv2d(net, 256, [3,3], scope='conv3_3')
      net = slim.conv2d(net, 256, [3,3], scope='conv3_4')
      net = slim.max_pool2d(net, scope='pool3_stage1')
      net = slim.conv2d(net, 512, [3,3], scope='conv4_1')
      
  with slim.arg_scope([slim.conv2d], stride=1, padding='SAME',
                      activation_fn=prelu, normalizer_fn=None):
    net = slim.conv2d(net, 512, [3,3], scope='conv4_2')
    net = slim.conv2d(net, 256, [3,3], scope='conv4_3_CPM')
    net = slim.conv2d(net, 128, [3,3], scope='conv4_4_CPM')
    end_points['stage0'] = net

    stage0_l2 = net
    for i in range(5):
      _0 = slim.conv2d(stage0_l2, 96, [3,3], scope='Mconv{}_stage0_L2_0'.format(i+1))
      _1 = slim.conv2d(_0, 96, [3,3], scope='Mconv{}_stage0_L2_1'.format(i+1))
      _2 = slim.conv2d(_1, 96, [3,3], scope='Mconv{}_stage0_L2_2'.format(i+1))
      stage0_l2 = tf.concat([_0, _1, _2], 3, name='Mconv{}_stage0_L2_concat'.format(i+1))
    stage0_l2 = slim.conv2d(stage0_l2, 256, [1,1], scope='Mconv6_stage0_L2')
    stage0_l2 = slim.conv2d(stage0_l2, num_limbs*2, [1,1], activation_fn=None, scope='Mconv7_stage0_L2')
    end_points['stage1_L1'] = stage0_l2

    stagej_l2 = stage0_l2
    for j in range(3):
      stagej_l2 = tf.concat([net, stagej_l2], 3, name='concat_stage{}_L2'.format(j+1))
      for i in range(5):
        _0 = slim.conv2d(stagej_l2, 128, [3,3], scope='Mconv{}_stage{}_L2_0'.format(i+1,j+1))
        _1 = slim.conv2d(_0, 128, [3,3], scope='Mconv{}_stage{}_L2_1'.format(i+1,j+1))
        _2 = slim.conv2d(_1, 128, [3,3], scope='Mconv{}_stage{}_L2_2'.format(i+1,j+1))
        stagej_l2 = tf.concat([_0, _1, _2], 3, name='Mconv{}_stage{}_L2_concat'.format(i+1,j+1))
      stagej_l2 = slim.conv2d(stagej_l2, 512, [1,1], scope='Mconv6_stage{}_L2'.format(j+1))
      stagej_l2 = slim.conv2d(stagej_l2, num_limbs*2, [1,1], activation_fn=None, scope='Mconv7_stage{}_L2'.format(j+1))
      end_points['stage{}_L1'.format(j+2)] = stagej_l2

    stage0_l1 = tf.concat([net, stagej_l2], 3, name='concat_stage0_L1')
    for i in range(5):
      _0 = slim.conv2d(stage0_l1, 96, [3,3], scope='Mconv{}_stage0_L1_0'.format(i+1))
      _1 = slim.conv2d(_0, 96, [3,3], scope='Mconv{}_stage0_L1_1'.format(i+1))
      _2 = slim.conv2d(_1, 96, [3,3], scope='Mconv{}_stage0_L1_2'.format(i+1))
      stage0_l1 = tf.concat([_0, _1, _2], 3, name='Mconv{}_stage0_L1_concat'.format(i+1))
    stage0_l1 = slim.conv2d(stage0_l1, 256, [1,1], scope='Mconv6_stage0_L1')
    stage0_l1 = slim.conv2d(stage0_l1, num_parts, [1,1], activation_fn=None, scope='Mconv7_stage0_L1')
    end_points['stage1_L2'] = stage0_l1
      
    stagej_l1 = stage0_l1
    for j in range(1):
      stagej_l1 = tf.concat([net, stagej_l1, stagej_l2], 3, name='concat_stage{}_L1'.format(j+1))
      for i in range(5):
        _0 = slim.conv2d(stagej_l1, 128, [3,3], scope='Mconv{}_stage{}_L1_0'.format(i+1,j+1))
        _1 = slim.conv2d(_0, 128, [3,3], scope='Mconv{}_stage{}_L1_1'.format(i+1,j+1))
        _2 = slim.conv2d(_1, 128, [3,3], scope='Mconv{}_stage{}_L1_2'.format(i+1,j+1))
        stagej_l1 = tf.concat([_0, _1, _2], 3, name='Mconv{}_stage{}_L1_concat'.format(i+1,j+1))
      stagej_l1 = slim.conv2d(stagej_l1, 512, [1,1], scope='Mconv6_stage{}_L1'.format(j+1))
      stagej_l1 = slim.conv2d(stagej_l1, num_parts, [1,1], activation_fn=None, scope='Mconv7_stage{}_L1'.format(j+1))
    end_points['stage{}_L2'.format(j+2)] = stagej_l1
  return end_points

def base_net(x, num_parts=71, num_stages=6):
  end_points = {}

  with slim.arg_scope([slim.max_pool2d], stride=2, kernel_size=[2,2]):
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME',
              activation_fn=tf.nn.relu, normalizer_fn=None):
      net = slim.conv2d(x, 64, [3,3], scope='conv1_1')
      net = slim.conv2d(net, 64, [3,3], scope='conv1_2')
      net = slim.max_pool2d(net, scope='pool1_stage1')
      net = slim.conv2d(net, 128, [3,3], scope='conv2_1')
      net = slim.conv2d(net, 128, [3,3], scope='conv2_2')
      net = slim.max_pool2d(net, scope='pool2_stage1')
      net = slim.conv2d(net, 256, [3,3], scope='conv3_1')
      net = slim.conv2d(net, 256, [3,3], scope='conv3_2')
      net = slim.conv2d(net, 256, [3,3], scope='conv3_3')
      net = slim.conv2d(net, 256, [3,3], scope='conv3_4')
      net = slim.max_pool2d(net, scope='pool3_stage1')
      net = slim.conv2d(net, 512, [3,3], scope='conv4_1')
      net = slim.conv2d(net, 512, [3,3], scope='conv4_2')
      net = slim.conv2d(net, 512, [3,3], scope='conv4_3')
      net = slim.conv2d(net, 512, [3,3], scope='conv4_4')
      net = slim.conv2d(net, 512, [3,3], scope='conv5_1')
      net = slim.conv2d(net, 512, [3,3], scope='conv5_2')
      net = slim.conv2d(net, 128, [3,3], scope='conv5_3_CPM')
      part = slim.conv2d(net, 512, [1,1], scope='conv6_1_CPM')
      part = slim.conv2d(part, num_parts, [1,1], activation_fn=None, scope='conv6_2_CPM')
      end_points['stage1'] = part

  with slim.arg_scope([slim.conv2d], stride=1, padding='SAME',
            activation_fn=tf.nn.relu, normalizer_fn=None):
    for stage in range(2, num_stages+1):
      concat = tf.concat(values=[part, net], axis=3, name='concat_stage{}'.format(stage))
      part = slim.conv2d(concat, 128, [7,7], scope='Mconv1_stage{}'.format(stage))
      part = slim.conv2d(part, 128, [7,7], scope='Mconv2_stage{}'.format(stage))
      part = slim.conv2d(part, 128, [7,7], scope='Mconv3_stage{}'.format(stage))
      part = slim.conv2d(part, 128, [7,7], scope='Mconv4_stage{}'.format(stage))
      part = slim.conv2d(part, 128, [7,7], scope='Mconv5_stage{}'.format(stage))
      part = slim.conv2d(part, 128, [1,1], scope='Mconv6_stage{}'.format(stage))
      part = slim.conv2d(part, num_parts, [1,1], activation_fn=None, scope='Mconv7_stage{}'.format(stage))
      end_points['stage{}'.format(stage)] = part

  return end_points

def face_net(x):
  return base_net(x, num_parts=71, num_stages=6)

def hand_net(x):
  return base_net(x, num_parts=22, num_stages=6)

# npy file must have been converted from caffemodel using caffe-tensorflow
def convert_npy_to_ckpt(net_fn, npy_path, ckpt_path):
  import numpy as np
  graph = tf.Graph()
  with graph.as_default():
    with tf.Session(graph=graph) as sess:
      npy = np.load(npy_path, encoding='latin1').item()
      x = tf.placeholder(tf.float32, shape=[None,None,None,3])
      assign_value = tf.placeholder(tf.float32)
      net_fn(x)
      variables = {v.name: v for v in tf.global_variables()}
      
      for op_name, op_data in npy.items():
        for param_name, param_data in op_data.items():
          var_name = op_name+'/'+param_name+':0'
          try:
            if 'prelu' in op_name:
              var_names = variables.keys()
              var_names = filter(
                lambda x: x.startswith(op_name.replace('prelu','conv')) \
                       and x.endswith('alpha:0'), var_names)
              var_name = list(var_names)[0]
            var = variables[var_name]
            sess.run(var.assign(assign_value), {assign_value: param_data})
          except:
            raise
            #rospy.logerr('{} is not in the network.'.format(var_name))
      tf.train.Saver(sharded=False).save(sess, ckpt_path, write_meta_graph=False)

def non_maximum_suppression(heat_map, threshold=0.5):
  if isinstance(heat_map, np.ndarray):
    global _max_pool_sess
    if '_max_pool_sess' not in globals():
      with tf.Graph().as_default() as graph:
        ph_x = tf.placeholder(tf.float32, shape=[None]*4, name='x')
        ph_t = tf.placeholder(tf.float32, shape=[], name='t')
        y = non_maximum_suppression(ph_x, threshold=ph_t)
        y = tf.identity(y, name='y')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
      _max_pool_sess = tf.Session(graph=graph, config=config)
    return _max_pool_sess.run('y:0', {'x:0': heat_map, 't:0': threshold})
  
  heat_map_max = slim.max_pool2d(
    heat_map, stride=1, kernel_size=[3,3], padding='SAME')
  # (num, 4(NHWC))
  inds = tf.where(
    tf.logical_and(heat_map > threshold,
                   tf.equal(heat_map_max, heat_map)))
  #return tf.concat(values=[inds[:,0:1], inds[:, 1:3] * 8, inds[:,3:]], axis=1)
  return inds

def connect_parts(affinity, keypoints, limbs, line_division=10, threshold=0.2):
  persons = [{c: id} for id, (_,_,c) in enumerate(keypoints)]
  for k, (p, q) in enumerate(limbs):
    is_p = keypoints[:,2] == p
    is_q = keypoints[:,2] == q
    p_inds = np.where(is_p)[0]
    q_inds = np.where(is_q)[0]
    q_mesh, p_mesh = np.meshgrid(np.where(is_q), np.where(is_p))
    Px = keypoints[p_mesh, 1]
    Py = keypoints[p_mesh, 0]
    Qx = keypoints[q_mesh, 1]
    Qy = keypoints[q_mesh, 0]
    Dx = Qx - Px
    Dy = Qy - Py
    norm = np.sqrt(Dx**2 + Dy**2)
    piled = norm==0.
    norm[piled] = 1
    Dx = Dx/norm
    Dy = Dy/norm
    Lx = np.zeros_like(Dx)
    Ly = np.zeros_like(Dy)
    for u in np.linspace(0,1,line_division):
      Rx = np.int32((1-u) * Px + u * Qx)
      Ry = np.int32((1-u) * Py + u * Qy)
      Lx += affinity[Ry,Rx,k*2]
      Ly += affinity[Ry,Rx,k*2+1]
    C = (Dx*Lx + Dy*Ly)/line_division
    C[piled] = threshold
    # rospy.loginfo('norm==0: {}'.format(np.where(norm==0)))
    # if k==10:
    #   rospy.loginfo(C)
    # I, J = scipy.optimize.linear_sum_assignment(-C)
    I, J = optimize.linear_sum_assignment(-C)
    for i, j in zip(I, J):
      if C[i,j] < threshold:
        continue
      i = p_inds[i]
      j = q_inds[j]
      matched = list(filter(lambda person: i in person.values(), persons))
      matched.extend(filter(lambda person: j in person.values(), persons))
      if len(matched) > 1:
        # rospy.loginfo('{}->{}: {} entries will be merged.'.format(i, j, len(matched)))
        merged = {}
        for person in matched:
          merged.update(person)
          if person in persons:
            persons.remove(person)
        # rospy.loginfo('{} -> {}'.format(matched, merged))
        persons.append(merged)
  #return [{k:keypoints[v,:2] for k,v in p.iteritems()} for p in persons]
  return persons
