import tensorflow as tf
import common

def inception(x, conv1_size, conv3_size, conv5_size, pool1_size, wd, is_training):
  with tf.variable_scope("conv_1"):
    conv1 = common.spatialConvolution(x, 1, 1, conv1_size, wd= wd)
    conv1 = common.batchNormalization(conv1, is_training= is_training)
    conv1 = tf.nn.relu(conv1)
  with tf.variable_scope("conv_3_1"):
    conv3 = common.spatialConvolution(x, 1, 1, conv3_size[0], wd= wd)
    conv3 = common.batchNormalization(conv3, is_training= is_training)
    conv3 = tf.nn.relu(conv3)
  with tf.variable_scope("conv_3_2"):
    conv3 = common.spatialConvolution(conv3, 3, 1, conv3_size[1], wd= wd)
    conv3 = common.batchNormalization(conv3, is_training= is_training)
    conv3 = tf.nn.relu(conv3)
  with tf.variable_scope("conv_5_1"):
    conv5 = common.spatialConvolution(x, 1, 1, conv5_size[0], wd= wd)
    conv5 = common.batchNormalization(conv5, is_training= is_training)
    conv5 = tf.nn.relu(conv5)
  with tf.variable_scope("conv_5_2"):
    conv5 = common.spatialConvolution(conv5, 5, 1, conv5_size[1], wd= wd)
    conv5 = common.batchNormalization(conv5, is_training= is_training)
    conv5 = tf.nn.relu(conv5)
  with tf.variable_scope("pool_1"):
    pool1= common.maxPool(x, 3, 1) 
    pool1 = common.spatialConvolution(pool1, 1, 1, pool1_size, wd= wd)
    pool1 = common.batchNormalization(pool1, is_training= is_training)
    pool1 = tf.nn.relu(pool1)
  return tf.concat([conv1, conv3, conv5, pool1], 3)

def inference(x, num_output, wd, dropout_rate, is_training, transfer_mode= False):
  with tf.variable_scope('features'):
    with tf.variable_scope('conv1'):
      network = common.spatialConvolution(x, 7, 2, 64, wd= wd)
      network = common.batchNormalization(network, is_training= is_training)
      network = tf.nn.relu (network)
    network = common.maxPool(network, 3, 2)
    with tf.variable_scope('conv2'):
      network = common.spatialConvolution(network, 1, 1, 64, wd= wd)
      network = common.batchNormalization(network, is_training= is_training)
      network = tf.nn.relu(network)
    with tf.variable_scope('conv3'):
      network = common.spatialConvolution(network, 3, 1, 192, wd= wd)
      network = common.batchNormalization(network, is_training= is_training)
      network = tf.nn.relu(network) 
    network = common.maxPool(network, 3, 2)
    with tf.variable_scope('inception3a'):
      network = inception( network, 64, [96, 128], [16, 32], 32, wd= wd, is_training= is_training)
    with tf.variable_scope('inception3b'):
      network = inception( network, 128, [128, 192], [32, 96], 64, wd= wd, is_training= is_training)
    network = common.maxPool(network, 3, 2)
    with tf.variable_scope('inception4a'):
      network = inception( network, 192, [96, 208], [16, 48], 64, wd= wd, is_training= is_training)
    with tf.variable_scope('inception4b'):
      network = inception( network, 160, [112, 224], [24, 64], 64, wd= wd, is_training= is_training)
    with tf.variable_scope('inception4c'):
      network = inception( network, 128, [128, 256], [24, 64], 64, wd= wd, is_training= is_training)
    with tf.variable_scope('inception4d'):
      network = inception( network, 112, [144, 288], [32, 64], 64, wd= wd, is_training= is_training)

  with tf.variable_scope('mainb'):
    with tf.variable_scope('inception4e'):
      main_branch = inception( network, 256, [160, 320], [32, 128], 128, wd= wd, is_training= is_training) 
    main_branch = common.maxPool(main_branch, 3, 2)
    with tf.variable_scope('inception5a'):
      main_branch= inception(main_branch, 256, [160, 320], [32, 128], 128, wd= wd, is_training= is_training)
    with tf.variable_scope('inception5b'):
      main_branch= inception(main_branch, 384, [192, 384], [48, 128], 128, wd= wd, is_training= is_training)
    main_branch= common.avgPool(main_branch, 7, 1)
    main_branch= common.flatten(main_branch)
    main_branch= tf.nn.dropout(main_branch, dropout_rate)
    if not transfer_mode:
      with tf.variable_scope('output'):
        main_branch= common.fullyConnected(main_branch, num_output, wd= wd)
    else:
      with tf.variable_scope('transfer_output'):
        main_branch= common.fullyConnected(main_branch, num_output, wd= wd)

  with tf.variable_scope('auxb'):
    aux_classifier= common.avgPool(network, 5, 3)
    with tf.variable_scope('conv1'):
      aux_classifier= common.spatialConvolution(aux_classifier, 1, 1, 128, wd= wd)
      aux_classifier= common.batchNormalization(aux_classifier, is_training= is_training)
      aux_classifier= tf.nn.relu(aux_classifier)
    aux_classifier= common.flatten(aux_classifier)
    with tf.variable_scope('fc1'):
      aux_classifier= common.fullyConnected(aux_classifier, 1024, wd= wd)
      aux_classifier= tf.nn.dropout(aux_classifier, dropout_rate)
    if not transfer_mode:
      with tf.variable_scope('output'):
        aux_classifier= common.fullyConnected(aux_classifier, num_output, wd= wd)
    else:
      with tf.variable_scope('transfer_output'):
        aux_classifier= common.fullyConnected(aux_classifier, num_output, wd= wd)
 
  return tf.concat([main_branch, aux_classifier],1)




