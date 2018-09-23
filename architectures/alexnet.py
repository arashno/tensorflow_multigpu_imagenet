import tensorflow as tf
from .common import *
from .model import *



#AlexNet architecture
def alexnet(net, num_classes, wd, dropout_rate, is_training):

  with tf.variable_scope('conv1'):
    net = spatialConvolution(net, 11, 4, 64, wd= wd)
    net = batchNormalization(net, is_training= is_training)
    net = tf.nn.relu (net)
    #common.activation_summary(net)

  net = maxPool(net, 3, 2)

  with tf.variable_scope('conv2'):
    net = spatialConvolution(net, 5, 1, 192, wd= wd)
    net = batchNormalization(net, is_training= is_training)
    net = tf.nn.relu(net)
    #common.activation_summary(net)

  net = maxPool(net, 3, 2)

  with tf.variable_scope('conv3'):
    net = spatialConvolution(net, 3, 1, 384, wd= wd)
    net = batchNormalization(net, is_training= is_training)
    net = tf.nn.relu(net)
    #common.activation_summary(net)

  with tf.variable_scope('conv4'):
    net = spatialConvolution(net, 3, 1, 256, wd= wd)
    net = batchNormalization(net, is_training= is_training)
    net = tf.nn.relu(net)

  with tf.variable_scope('conv5'):
    net = spatialConvolution(net, 3, 1, 256, wd= wd)
    net = batchNormalization(net, is_training= is_training)
    net = tf.nn.relu(net)

  net = maxPool(net, 3, 2)

  net = flatten(net)

  with tf.variable_scope('fc1'): 
    net = tf.nn.dropout(net, dropout_rate)
    net = fullyConnected(net, 4096, wd= wd)
    net = batchNormalization(net, is_training= is_training)
    net = tf.nn.relu(net)

  with tf.variable_scope('fc2'):
    net = tf.nn.dropout(net, dropout_rate)
    net = fullyConnected(net, 4096, wd= wd)
    net = batchNormalization(net, is_training= is_training)
    net = tf.nn.relu(net)

  with tf.variable_scope('output'):
    net = fullyConnected(net, num_classes, wd= wd)

  return net
