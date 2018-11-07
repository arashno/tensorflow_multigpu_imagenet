from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np

import tensorflow as tf

import sys
import argparse

from architectures.common import SAVE_VARIABLES

"""
A simple utility class for computing averages of loss and accuracies
"""
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

"""
This methods counts the number of examples in an input file and calculates the number of batches for each epoch.
Args:
    filename: the name of the input file
    batch_size: batch size
Returns:
    number of samples and number of batches
"""
def count_input_records(filename, batch_size):
  with open(filename) as f:
    num_samples = sum(1 for line in f)
  num_batches=num_samples/batch_size
  return num_samples, int(num_batches) if num_batches.is_integer() else int(num_batches)+1

"""
  Compute cross-entropy loss for the given logits and labels
  Add summary for the cross entropy loss

  Args:
    logits: Logits from the model
    labels: Labels from data_loader
  Returns:
    Loss tensor of type float.
"""
def loss(logits, labels):
  # Calculate the average cross entropy loss across the batch.
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= labels, logits= logits, name= 'cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name= 'cross_entropy')

  #Add a Tensorboard summary
  tf.summary.scalar('Cross Entropy Loss', cross_entropy_mean)

  return cross_entropy_mean

"""
This methods parses an input string to determine details of a learning rate policy.

Args:
    policy_type: Type of the policy
    details_str: the string to parse
"""
def get_policy(policy_type, details_str):
  if policy_type=='constant':
    return tf.constant(float(details_str))
  if policy_type=='piecewise_linear':
    details= [float(x) for x in details_str.split(",")]
    length = len(details)
    assert length%2==1, 'Invalid policy details'
    assert all(item.is_integer() for item in details[0:int((length-1)/2)]), 'Invalid policy details'
    return tf.train.piecewise_constant(tf.get_collection(SAVE_VARIABLES, scope="epoch_number")[0], [int(x) for x in details[0:int((length-1)/2)]], details[int((length-1)/2):])
  if policy_type=='exponential':
    details= [float(x) for x in details_str.split(',')]
    assert details[1].is_integer(), 'Invalid policy details'
    return tf.train.exponential_decay(details[0], tf.get_collection(SAVE_VARIABLES, scope='global_step')[0], int(details[1]), details[2], staircase=False)


"""
this method return an instance of a type of optimization algorithms based on the arguments.
Args:
    opt_type: type of the algorithm
    lr: learning rate policy
"""
def get_optimizer(opt_type, lr):
  if opt_type.lower()=='momentum':
    return tf.train.MomentumOptimizer(lr, 0.9)
  elif opt_type.lower()=='adam':
    return tf.train.AdamOptimizer()
  elif opt_type.lower()=='adadelta':
    return tf.train.AdadeltaOptimizer()
  elif opt_type.lower()=='adagrad':
    return tf.train.AdagradOptimizer(lr)
  elif opt_type.lower()=='rmsprop':
    return tf.train.RMSPropOptimizer(lr)
  elif opt_type.lower()=='sgd':
    return tf.train.GradientDescentOptimizer(lr)
  else:
    print("invalid optimizer")
    return None
