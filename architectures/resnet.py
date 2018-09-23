import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from .common import *
from .model import *

# ResNet Stack
def resnetStack(net, num_blocks, stack_stride, block_filters_internal, bottleneck, wd= 0.0, is_training= True):

  for n in range(num_blocks):
    s = stack_stride if n == 0 else 1
    block_stride = s

    with tf.variable_scope('block%d' % (n + 1)):
      net = resnetBlock(net, bottleneck, block_filters_internal, block_stride, wd= wd, is_training= is_training)

  return net

# ResNet Block
def resnetBlock(net, bottleneck, block_filters_internal, block_stride, wd= 0.0, is_training= True):

  filters_in = net.get_shape()[-1]

  # Note: filters_out isn't how many filters are outputed.
  # That is the case when bottleneck=False but when bottleneck is
  # True, filters_internal*4 filters are outputted. filters_internal is how many filters
  # the 3x3 convs output internally.
  m = 4 if bottleneck else 1
  filters_out = m * block_filters_internal

  shortcut = net  # branch 1

  conv_filters_out = block_filters_internal

  conv_weight_initializer = tf.truncated_normal_initializer(stddev= 0.1)

  if bottleneck:

    with tf.variable_scope('a'):
      net = spatialConvolution(net, 1, block_stride, conv_filters_out, weight_initializer= conv_weight_initializer, wd= wd)
      net = batchNormalization(net, is_training= is_training)
      net = tf.nn.relu(net)

    with tf.variable_scope('b'):
      net = spatialConvolution(net, 3, 1, conv_filters_out, weight_initializer= conv_weight_initializer, wd= wd)
      net = batchNormalization(net, is_training= is_training)
      net = tf.nn.relu(net)

    with tf.variable_scope('c'):
      net = spatialConvolution(net, 1, 1, filters_out, weight_initializer= conv_weight_initializer, wd= wd)
      net = batchNormalization(net, is_training= is_training)

  else:

    with tf.variable_scope('A'):
      net = spatialConvolution(net, 3, block_stride, conv_filters_out, weight_initializer= conv_weight_initializer, wd= wd)
      net = batchNormalization(net, is_training= is_training)
      net = tf.nn.relu(net)

    with tf.variable_scope('B'):
      net = spatialConvolution(net, 3, 1, filters_out, weight_initializer= conv_weight_initializer, wd= wd)
      net = batchNormalization(net, is_training= is_training)

  with tf.variable_scope('shortcut'):

    if filters_out != filters_in or block_stride != 1:
      shortcut = spatialConvolution(shortcut, 1, block_stride, filters_out, weight_initializer= conv_weight_initializer, wd= wd)
      shortcut = batchNormalization(shortcut, is_training= is_training)

  return tf.nn.relu(net + shortcut)



# Build the model based on the depth arg
def resnet(net, num_classes, wd, is_training, transfer_mode, depth):

  num_blockes= []

  bottleneck= False

  if depth == 18:
    num_blocks= [2, 2, 2, 2]

  elif depth == 34:
    num_blocks= [3, 4, 6, 3]

  elif depth == 50:
    num_blocks= [3, 4, 6, 3]
    bottleneck= True

  elif depth == 101:
    num_blocks= [3, 4, 23, 3]
    bottleneck= True

  elif depth == 152:
    num_blocks= [3, 8, 36, 3]
    bottleneck= True

  return getModel(net, num_classes, wd, is_training, num_blocks= num_blocks, bottleneck= bottleneck, transfer_mode= transfer_mode)

# a helper function to have a more organized code
def getModel(net, num_output, wd, is_training, num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
            bottleneck= True, transfer_mode= False):

  conv_weight_initializer = tf.truncated_normal_initializer(stddev= 0.1)

  fc_weight_initializer = tf.truncated_normal_initializer(stddev= 0.01)

  with tf.variable_scope('scale1'):
    net = spatialConvolution(net, 7, 2, 64, weight_initializer= conv_weight_initializer, wd= wd)
    net = batchNormalization(net, is_training= is_training)
    net = tf.nn.relu(net)

  with tf.variable_scope('scale2'):
    net = maxPool(net, 3, 2)
    net = resnetStack(net, num_blocks[0], 1, 64, bottleneck, wd= wd, is_training= is_training)

  with tf.variable_scope('scale3'):
    net = resnetStack(net, num_blocks[1], 2, 128, bottleneck, wd= wd, is_training= is_training)

  with tf.variable_scope('scale4'):
    net = resnetStack(net, num_blocks[2], 2, 256, bottleneck, wd= wd, is_training= is_training)

  with tf.variable_scope('scale5'):
    net = resnetStack(net, num_blocks[3], 2, 512, bottleneck, wd= wd, is_training= is_training)

  # post-net
  net = tf.reduce_mean(net, reduction_indices= [1, 2], name= "avg_pool")

  with tf.variable_scope('output'):
    net = fullyConnected(net, num_output, weight_initializer= fc_weight_initializer, bias_initializer= tf.zeros_initializer, wd= wd)

  return net
