from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.estimator import regression
from tflearn.activations import relu

def block(a,b,c,x):
      no = a
      network = conv_2d(x, a, b, strides=c)
      network = batch_normalization(network,epsilon=0.001)
      network = relu (network)
      network = conv_2d(network, a,1,strides=1)
      network = batch_normalization(network,epsilon=0.001)
      network = relu (network)
      network = conv_2d(network, a, 1, strides=1)
      network = batch_normalization(network,epsilon=0.001)
      network = relu (network)
      return network

def mp(a,b,x):
      network = max_pool_2d(x, a, strides=b)
      return network

def nin(x,num_classes):
   network= block(96, 11, 4, x)
   network= mp(3, 2, network)
   network= block(256, 5, 1, network)
   network= mp(3, 2, network)
   network= block(384, 3,1, network)
   network= mp(3, 2, network)
   network= block(1024, 3, 1, network)

   network= avg_pool_2d(network,7)

   network = fully_connected(network, num_classes)

   return network



