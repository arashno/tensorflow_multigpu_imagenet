import tensorflow as tf
import re
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from math import sqrt

RESNET_VARIABLES = 'resnet_variables'
TOWER_NAME = 'Tower'

def _get_variable(name,
                  shape,
                  initializer,
                  regularizer= None,
                  dtype= 'float',
                  trainable= True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
    with tf.device('/cpu:0'):
      var = tf.get_variable(name,
                           shape= shape,
                           initializer= initializer,
                           dtype= dtype,
                           regularizer= regularizer,
                           collections= collections,
                           trainable= trainable)

    return var

def batchNormalization(x, is_training= True, decay= 0.9, epsilon= 0.001):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]


    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer= tf.zeros_initializer)
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer= tf.ones_initializer)

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer= tf.zeros_initializer,
                                trainable= False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer= tf.ones_initializer,
                                    trainable= False)

    # These ops will only be preformed when training.

    if is_training:
      mean, variance = tf.nn.moments(x, axis)
      update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, decay)
      update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, decay)
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS , update_moving_mean)
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS , update_moving_variance)
      return tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)
    else:
      return tf.nn.batch_normalization(x, moving_mean, moving_variance, beta, gamma, epsilon)


def flatten(x):
    shape = x.get_shape().as_list()
    dim = 1
    for i in xrange(1,len(shape)):
      dim*=shape[i]
    return tf.reshape(x, [-1, dim])

def treshold(x, treshold):
    return tf.cast(x > treshold, x.dtype) * x

def fullyConnected(x, num_units_out, wd= 0.0, weight_initializer= None, bias_initializer= None):
    num_units_in = x.get_shape()[1]

    stddev = 1./tf.sqrt(tf.cast(num_units_out, tf.float32))
    if weight_initializer is None:
      weight_initializer = tf.random_uniform_initializer(minval= -stddev, maxval= stddev, dtype= tf.float32)
    if bias_initializer is None:
      bias_initializer = tf.random_uniform_initializer(minval= -stddev, maxval= stddev, dtype= tf.float32) 

    weights = _get_variable('weights',
                            [num_units_in, num_units_out], weight_initializer, tf.contrib.layers.l2_regularizer(wd))

    biases = _get_variable('biases',
                           [num_units_out], bias_initializer)
                           
    return tf.nn.xw_plus_b(x, weights, biases)

def spatialConvolution(x, ksize, stride, filters_out, wd= 0.0, weight_initializer= None, bias_initializer= None):
    filters_in = x.get_shape()[-1]
    stddev = 1./tf.sqrt(tf.cast(filters_out, tf.float32))
    if weight_initializer is None:
      weight_initializer = tf.random_uniform_initializer(minval= -stddev, maxval= stddev, dtype= tf.float32)
    if bias_initializer is None:
      bias_initializer = tf.random_uniform_initializer(minval= -stddev, maxval= stddev, dtype= tf.float32) 

    shape = [ksize, ksize, filters_in, filters_out]
    weights = _get_variable('weights',
                            shape, weight_initializer, tf.contrib.layers.l2_regularizer(wd))

    conv = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding= 'SAME')
    biases = _get_variable('biases', [filters_out],  bias_initializer)
            
    return tf.nn.bias_add(conv, biases)
    
def maxPool(x, ksize, stride):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')

def avgPool(x, ksize, stride, padding='SAME'):
    return tf.nn.avg_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding= padding)
    
def resnetStack(x, num_blocks, stack_stride, block_filters_internal, bottleneck, wd= 0.0, is_training= True):
    for n in range(num_blocks):
        s = stack_stride if n == 0 else 1
        block_stride = s
        with tf.variable_scope('block%d' % (n + 1)):
            x = resnetBlock(x, bottleneck, block_filters_internal, block_stride, wd= wd, is_training= is_training)
    return x


def resnetBlock(x, bottleneck, block_filters_internal, block_stride, wd= 0.0, is_training= True):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed. 
    # That is the case when bottleneck=False but when bottleneck is 
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    m = 4 if bottleneck else 1
    filters_out = m * block_filters_internal

    shortcut = x  # branch 1

    conv_filters_out = block_filters_internal

    
    conv_weight_initializer = tf.truncated_normal_initializer(stddev= 0.1)

    if bottleneck:
        with tf.variable_scope('a'):
            x = spatialConvolution(x, 1, block_stride, conv_filters_out, weight_initializer= conv_weight_initializer, wd= wd)
            x = batchNormalization(x, is_training= is_training)
            x = tf.nn.relu(x)

        with tf.variable_scope('b'):
            x = spatialConvolution(x, 3, 1, conv_filters_out, weight_initializer= conv_weight_initializer, wd= wd)
            x = batchNormalization(x, is_training= is_training)
            x = tf.nn.relu(x)

        with tf.variable_scope('c'):
            x = spatialConvolution(x, 1, 1, filters_out, weight_initializer= conv_weight_initializer, wd= wd)
            x = batchNormalization(x, is_training= is_training)
    else:
        with tf.variable_scope('A'):
            x = spatialConvolution(x, 3, block_stride, conv_filters_out, weight_initializer= conv_weight_initializer, wd= wd)
            x = batchNormalization(x, is_training= is_training)
            x = tf.nn.relu(x)

        with tf.variable_scope('B'):
            x = spatialConvolution(x, 1, 1, filters_out, weight_initializer= conv_weight_initializer, wd= wd)
            x = batchNormalization(x, is_training= is_training)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or block_stride != 1:
            shortcut = spatialConvolution(shortcut, 1, block_stride, filters_out, weight_initializer= conv_weight_initializer, wd= wd)
            shortcut = batchNormalization(shortcut, is_training= is_training)

    return tf.nn.relu(x + shortcut)


