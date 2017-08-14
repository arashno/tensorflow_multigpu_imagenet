"""A program to train different architectures(AlexNet,ResNet,...) using multiple GPU's with synchronous updates.

Usage:
Please refer to the readme file to compile the program and train the model.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange

import tensorflow as tf

import data_loader
import arch
import sys
import argparse


def loss(logits, labels):
  """
  Compute cross-entropy loss for the given logits and labels
  Add summary for the cross entropy loss

  Args:
    logits: Logits from the model
    labels: Labels from data_loader
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= labels, logits= logits, name= 'cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name= 'cross_entropy')

  #Add a Tensorboard summary
  tf.summary.scalar('Cross Entropy Loss', cross_entropy_mean)

  return cross_entropy_mean


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all the GPUs.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis= 0, values= grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train(args):
  """Train different architectures for a number of epochs."""

  with tf.Graph().as_default(), tf.device('/cpu:0'):

    # Read data from disk
    images, labels = data_loader.read_inputs(True, args)

    epoch_number = tf.get_variable('epoch_number', [], dtype= tf.int32, initializer= tf.constant_initializer(0), trainable= False)

    # Decay the learning rate
    lr = tf.train.piecewise_constant(epoch_number, [19, 30, 44, 53], [
                                     0.01, 0.005, 0.001, 0.0005, 0.0001], name= 'LearningRate')
    # Weight Decay policy
    wd = tf.train.piecewise_constant(
        epoch_number, [30], [0.0005, 0.0], name= 'WeightDecay')

    # Create an optimizer that performs gradient descent.
    opt = tf.train.MomentumOptimizer(lr, 0.9)

    # Calculate the gradients for each model tower.
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(args.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('Tower_%d' % i) as scope:
            # Calculate the loss for one tower. This function
            # constructs the entire model but shares the variables across
            # all towers.
            logits = arch.get_model(images, wd, True, args)
            
            # Top-1 accuracy
            top1acc = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))
            # Top-5 accuracy
            top5acc = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32))

            # Build the portion of the Graph calculating the losses. Note that we will
            # assemble the total_loss using a custom function below.
            cross_entropy_mean = loss(logits, labels)

            # Get all the regularization lesses and add them
            regularization_losses = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)
            
            reg_loss = tf.add_n(regularization_losses)

            #Add a tensorboard summary
            tf.summary.scalar('Regularization Loss', reg_loss)

            # Compute the total loss (cross entropy loss + regularization loss)
            total_loss = tf.add(cross_entropy_mean, reg_loss)

            # Attach a scalar summary for the total loss and top-1 and top-5 accuracies
            tf.summary.scalar('Total Loss', total_loss)
            tf.summary.scalar('Top-1 Accuracy', top1acc)
            tf.summary.scalar('Top-5 Accuracy', top5acc)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            # Gather batch normaliziation update operations
            batchnorm_updates = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope)

            # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = opt.compute_gradients(total_loss)
            # Keep track of the gradients across all towers.
            tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate and weight decay
    summaries.append(tf.summary.scalar('learning_rate', lr))
    summaries.append(tf.summary.scalar('weight_decay', wd))

    # Group all updates to into a single train op.
    #with tf.control_dependencies(bn_update_ops):
    apply_gradient_op = opt.apply_gradients(grads)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep= args.num_epochs)

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Logging the runtime information if requested
    if args.log_debug_info:
      run_options = tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
    else:
      run_options = None
      run_metadata = None

    # Creating a session to run the built graph
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement= True, 
        log_device_placement= args.log_device_placement))

    # Continue training from a saved snapshot
    if args.retrain_from is not None:
      saver.restore(sess, args.retrain_from)
    else:
      sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess= sess)
    
    # Setup a summary writer
    summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph)

    # Set the start epoch number
    start_epoch = sess.run(epoch_number + 1)

    # The main training loop
    for epoch in xrange(start_epoch, start_epoch + args.num_epochs):
      # update epoch_number
      sess.run(epoch_number.assign(epoch))

      # Trainig batches
      for step in xrange(args.num_batches):
    
        start_time = time.time()
        _, loss_value, top1_accuracy, top5_accuracy = sess.run(
            [train_op, cross_entropy_mean, top1acc, top5acc], options= run_options, run_metadata= run_metadata)
        duration = time.time() - start_time

        # Check for errors
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        # Logging and writing tensorboard summaries
        if step % 10 == 0:
          num_examples_per_step = args.chunked_batch_size * args.num_gpus
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = duration / args.num_gpus

          format_str = ('%s: epoch %d, step %d, loss = %.2f, Top-1 = %.2f Top-5 = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), epoch, step, loss_value, top1_accuracy, top5_accuracy,
                               examples_per_sec, sec_per_batch))
          sys.stdout.flush()
        if step % 100 == 0:
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(
              summary_str, args.num_batches * epoch + step)
          if args.log_debug_info:
            summary_writer.add_run_metadata(
                run_metadata, 'epoch%d step%d' % (epoch, step))

      # Save the model checkpoint periodically after each training epoch
      checkpoint_path = os.path.join(args.log_dir, args.snapshot_prefix)
      saver.save(sess, checkpoint_path, global_step= epoch)


def main():  # pylint: disable=unused-argument
    parser = argparse.ArgumentParser(description='Process Command-line Arguments')
    parser.add_argument('--load_size', nargs= 2, default= [256,256], type= int, action= 'store', help= 'The width and height of images for loading from disk')
    parser.add_argument('--crop_size', nargs= 2, default= [224,224], type= int, action= 'store', help= 'The width and height of images after random cropping')
    parser.add_argument('--batch_size', default= 128, type= int, action= 'store', help= 'The training batch size')
    parser.add_argument('--num_classes', default= 1000 , type=int, action='store', help= 'The number of classes')
    parser.add_argument('--num_channels', default= 3 , type= int, action= 'store', help= 'The number of channels in input images')
    parser.add_argument('--num_epochs', default= 55, type= int, action= 'store', help= 'The number of epochs')
    parser.add_argument('--path_prefix', default= './', action='store', help= 'the prefix address for images')
    parser.add_argument('--data_info', default= 'train.txt', action= 'store', help= 'Name of the file containing addresses and labels of training images')
    parser.add_argument('--shuffle', default= True, type= bool, action= 'store',help= 'Shuffle training data or not')
    parser.add_argument('--num_threads', default= 20, type= int, action='store', help= 'The number of threads for loading data')
    parser.add_argument('--log_dir', default= None, action= 'store', help= 'Path for saving Tensorboard info and checkpoints')
    parser.add_argument('--snapshot_prefix', default= 'snapshot', action= 'store', help= 'Prefix for checkpoint files')
    parser.add_argument('--architecture', default= 'resnet', help= 'The DNN architecture')
    parser.add_argument('--depth', default= 50, type= int, action= 'store', help= 'The depth of ResNet architecture')
    parser.add_argument('--run_name', default= 'Run'+str(time.strftime("-%d-%m-%Y_%H-%M-%S")), action= 'store', help= 'Name of the experiment')
    parser.add_argument('--num_gpus', default= 1, type= int, action= 'store', help= 'Number of GPUs')
    parser.add_argument('--log_device_placement', default= False, type= bool, help= 'Whether to log device placement or not')
    parser.add_argument('--delimiter', default= ' ', action= 'store', help= 'Delimiter of the input files')
    parser.add_argument('--retrain_from', default= None, action= 'store', help= 'Continue Training from a snapshot file')
    parser.add_argument('--log_debug_info', default= False, action= 'store', help= 'Logging runtime and memory usage info')
    parser.add_argument('--num_batches', default= -1, type= int, action= 'store', help= 'The number of batches per epoch')
    args = parser.parse_args()

    # Spliting examples between different GPUs
    args.chunked_batch_size = int(args.batch_size/args.num_gpus)

    # COunting number of training examples
    args.num_samples = sum(1 for line in open(args.data_info))

    # set the number of batches per epoch
    if args.num_batches==-1:
        args.num_batches= int(args.num_samples/args.batch_size)+1

    # creating the logging directory
    if args.log_dir is None:
      args.log_dir= args.architecture+"_"+args.run_name
    print(args)
    print("Saving everything in "+args.log_dir)

    if tf.gfile.Exists(args.log_dir):
      tf.gfile.DeleteRecursively(args.log_dir)
    tf.gfile.MakeDirs(args.log_dir)
 
    train(args)


if __name__ == '__main__':
  main()
