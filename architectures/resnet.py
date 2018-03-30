import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import common

# Build the model based on the depth arg
def inference(x, depth, num_output, wd, is_training, transfer_mode= False):
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

    return getModel(x, num_output, wd, is_training, num_blocks= num_blocks, bottleneck= bottleneck, transfer_mode= transfer_mode)

# a helper function to have a more organized code
def getModel(x, num_output, wd, is_training, num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
              bottleneck= True, transfer_mode= False):
    conv_weight_initializer = tf.truncated_normal_initializer(stddev= 0.1)
    fc_weight_initializer = tf.truncated_normal_initializer(stddev= 0.01)
    with tf.variable_scope('scale1'):
        x = common.spatialConvolution(x, 7, 2, 64, weight_initializer= conv_weight_initializer, wd= wd)
        x = common.batchNormalization(x, is_training= is_training)
        x = tf.nn.relu(x)

    with tf.variable_scope('scale2'):
        x = common.maxPool(x, 3, 2)
        x = common.resnetStack(x, num_blocks[0], 1, 64, bottleneck, wd= wd, is_training= is_training)

    with tf.variable_scope('scale3'):
        x = common.resnetStack(x, num_blocks[1], 2, 128, bottleneck, wd= wd, is_training= is_training)

    with tf.variable_scope('scale4'):
        x = common.resnetStack(x, num_blocks[2], 2, 256, bottleneck, wd= wd, is_training= is_training)

    with tf.variable_scope('scale5'):
        x = common.resnetStack(x, num_blocks[3], 2, 512, bottleneck, wd= wd, is_training= is_training)

    # post-net
    x = tf.reduce_mean(x, reduction_indices= [1, 2], name= "avg_pool")

    if not transfer_mode:
      with tf.variable_scope('output'):
        x = common.fullyConnected(x, num_output, weight_initializer= fc_weight_initializer, bias_initializer= tf.zeros_initializer, wd= wd)
    else:
      with tf.variable_scope('transfer_output'):
        x = common.fullyConnected(x, num_output, weight_initializer= fc_weight_initializer, bias_initializer= tf.zeros_initializer, wd= wd)

    return x
