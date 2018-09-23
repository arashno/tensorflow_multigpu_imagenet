import tensorflow as tf
from .common import *

# The VGG architecture, the default type is 'A'
def vgg(net, num_output, wd, dropout_rate, is_training, model_type= 'D'):
   # Create tables describing VGG configurations A, B, D, E
   if model_type == 'A':
      config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

   elif model_type == 'B':
      config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

   elif model_type == 'D':
      config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

   elif model_type == 'E':
      config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

   else:
      print('Unknown model type: ' + model_type + ' | Please specify a modelType A or B or D or E')

   for k,v in enumerate(config):
     if v == 'M':

       net= maxPool(net, 2, 2)

     else:  

       with tf.variable_scope('conv'+str(k)):
         net = spatialConvolution(net, 3, 1, v, wd= wd)
         net = batchNormalization(net, is_training= is_training)
         net = tf.nn.relu(net)

   net= flatten(net)

   with tf.variable_scope('fc1'): 
     net = fullyConnected(net, 4096, wd= wd)
     net = tf.nn.relu(net)
     net = tf.nn.dropout(net, dropout_rate)

   with tf.variable_scope('fc2'):
     net = fullyConnected(net, 4096, wd= wd)
     net = tf.nn.relu(net)
     net = tf.nn.dropout(net, dropout_rate)
   
   with tf.variable_scope('output'):
     net = fullyConnected(net, num_output, wd= wd)

   return net
