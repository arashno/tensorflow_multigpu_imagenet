import tensorflow as tf
from .common import *

# Sub-network Block
def block(net, spec, wd, is_training):

  with tf.variable_scope('conv1'):
    nin = spatialConvolution(net, spec[0], spec[1], spec[2], wd= wd)
    nin = batchNormalization(nin, is_training= is_training)
    nin = tf.nn.relu(nin)

  with tf.variable_scope('conv2'):
    nin = spatialConvolution(nin, 1, 1, spec[2], wd= wd)
    nin = batchNormalization(nin, is_training= is_training)
    nin = tf.nn.relu(nin)

  with tf.variable_scope('conv3'):
    nin = spatialConvolution(nin, 1, 1, spec[2], wd= wd)
    nin = batchNormalization(nin, is_training= is_training)
    nin = tf.nn.relu(nin)

  return nin

# NiN architecture
def nin(net, num_output, wd, is_training, transfer_mode= False):

    with tf.variable_scope('block1'):
      net = block(net, [11, 4, 96], wd, is_training)

    net = maxPool(net, 3, 2)

    with tf.variable_scope('block2'):
      net = block(net, [5, 1, 256], wd, is_training)

    net = maxPool(net, 3, 2)

    with tf.variable_scope('block3'):
      net = block(net, [3, 1, 384], wd, is_training)

    net = maxPool(net, 3, 2)

    with tf.variable_scope('block4'):
      net = block(net, [3, 1, 1024], wd, is_training)

    net = avgPool(net, 7, 1)

    net = flatten(net)

    with tf.variable_scope('output'):
        net = fullyConnected(net, num_output, wd= wd)

    return net
