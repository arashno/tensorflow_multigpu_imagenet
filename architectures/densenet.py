import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from .common import *
import math


def densenet(net, depth, num_output, wd, is_training):

    stages= []

    K=32

    if depth == 121:
      stages= [6, 12, 24, 16]

    elif depth == 169:
      stages= [6, 12, 32, 32]

    elif depth == 201:
      stages= [6, 12, 48, 32]

    elif depth == 161:
      stages= [6, 12, 36, 24]
      K= 48

    return getModel(net, num_output, K, stages, wd, is_training)

def full_conv(net, K, is_training, wd):

  with tf.variable_scope('conv1x1'):
    net = batchNormalization(net, is_training=is_training)
    net = tf.nn.relu(net)
    net = spatialConvolution(net, 1, 1, 4*K, wd=wd)

  with tf.variable_scope('conv3x3'):
    net = batchNormalization(net, is_training=is_training)
    net = tf.nn.relu(net)
    net = spatialConvolution(net, 3, 1, K, wd=wd)

  return net

def block(net, layers, K, is_training, wd):

  for idx in range(layers):
    with tf.variable_scope('L'+str(idx)):
      tmp = full_conv(net, K, is_training, wd= wd)
      net = tf.concat((net, tmp),3)

  return net

def transition(net, K, wd, is_training):

  with tf.variable_scope('conv'):
    net = batchNormalization(net, is_training=is_training)
    net = tf.nn.relu(net)
    shape = net.get_shape().as_list()
    dim = math.floor(shape[3]*0.5)
    net = spatialConvolution(net, 1, 1, dim, wd=wd)
    net = avgPool(net, 2, 2)

  return net

def getModel(net, num_output, K, stages, wd, is_training):

    with tf.variable_scope('conv1'):
        net = spatialConvolution(net, 7, 2, 2*K, wd= wd)
        net = batchNormalization(net, is_training= is_training)
        net = tf.nn.relu(net)
        net = maxPool(net, 3, 2)
        
    with tf.variable_scope('block1'):
        net = block(net, stages[0], K, is_training= is_training,  wd= wd)

    with tf.variable_scope('trans1'):
        net = transition(net, K, wd= wd, is_training= is_training)    

    with tf.variable_scope('block2'):
        net = block(net, stages[1], K, is_training= is_training, wd= wd)

    with tf.variable_scope('trans2'):
        net = transition(net, K, wd= wd, is_training= is_training)    

    with tf.variable_scope('block3'):
        net = block(net, stages[2], K, is_training= is_training, wd= wd)

    with tf.variable_scope('trans3'):
        net = transition(net, K, wd= wd, is_training= is_training)    

    with tf.variable_scope('block4'):
        net = block(net, stages[3], K, is_training= is_training, wd= wd)

    net = avgPool(net,7,1, padding='VALID')

    net= flatten(net)

    with tf.variable_scope('output'):
      net = fullyConnected(net, num_output, wd= wd)

    return net
