from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os

import numpy as np
import tensorflow as tf
import argparse
import arch
import data_loader
import sys

from architectures import common

def exclude():
  var_list = tf.global_variables()
  to_remove = []
  for var in var_list:
      if(var.name.find("transfer_output")>=0):
        to_remove.append(var)
  for x in to_remove:
      var_list.remove(x)
  return var_list


def transfer(args):
  
  with tf.Graph().as_default() as g, tf.device('/cpu:0'):
    # Read images and labels from disk
    images, labels = data_loader.read_inputs(True, args)
    # Epoch number
    epoch_number = tf.get_variable('epoch_number', [], dtype= tf.int32, initializer= tf.constant_initializer(0), trainable= False)
    # Learning rate policy
    lr = tf.train.piecewise_constant(epoch_number, [3,6], [0.001,0.0005,0.0001], name= 'LearningRate')
   
    opt = tf.train.MomentumOptimizer(lr, 0.9)

    with tf.device('/gpu:0'):
      # Load model for transfer learning
      if(args.finetune):
        # Finetune the model
        logits = arch.get_model(images, 0.0, True, args, transferMode= True)
      else:
        # Use the model as a fixed feature extractor
        logits = arch.get_model(images, 0.0, False, args, transferMode= True)
      
      # Cross entropy loss for the target dataset
      transfer_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels= labels, logits= logits))

      # Building train and update operations    
      if args.finetune:
        batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        train_op = tf.group(opt.minimize(transfer_loss), batchnorm_updates_op)
      else:
        train_op = opt.minimize(transfer_loss,var_list= tf.get_collection(tf.GraphKeys.VARIABLES, scope= 'transfer_output'))

    # Top-1 accuracy
    top1acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))
    # Top-5 accuracy
    top5acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32))
     
    # a loader for loading the pretrained model (it does not load the last layer) 
    pretrained_loader = tf.train.Saver(var_list= exclude())
    # a saver for saving transferred models
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Summary writer
    summary_writer = tf.summary.FileWriter(args.log_dir, g)

    # Logging runtime information
    if args.log_debug_info:
      run_options = tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
    else:
      run_options = None
      run_metadata = None

    with tf.Session(config=tf.ConfigProto( allow_soft_placement= True)) as sess:

      # Intializing the variables
      sess.run(tf.global_variables_initializer())

      # Load pretrained model
      ckpt = tf.train.get_checkpoint_state(args.load_pretrained_dir)
      
      # Restore graph variables
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        pretrained_loader.restore(sess, ckpt.model_checkpoint_path)
      else:
        return

      # Start the queue runners.
      coord = tf.train.Coordinator()
      
      threads = tf.train.start_queue_runners(sess= sess, coord = coord)

      for epoch in xrange(1, args.num_epochs+1):
        sess.run(epoch_number.assign(epoch))
        for step in xrange(0, args.num_batches):
          start_time = time.time()
          top1_accuracy, top5_accuracy, loss_value, _ = sess.run([top1acc, top5acc, transfer_loss, train_op], options= run_options, run_metadata= run_metadata)
          duration = time.time() - start_time
          
          assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

          if step % 10 == 0:
            num_examples_per_step = args.chunked_batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = duration

            format_str = ('%s: epoch %d, step %d, loss = %.2f, Top-1 = %.2f Top-5 = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
            print (format_str % (datetime.now(), epoch, step, loss_value, top1_accuracy, top5_accuracy,
                             examples_per_sec, sec_per_batch))
            sys.stdout.flush()
          if step % 100 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, args.num_batches*epoch+step)
            if args.log_debug_info:
              summary_writer.add_run_metadata(run_metadata, 'epoch%d step%d' % (epoch, step))
 
        checkpoint_path = os.path.join(args.log_dir, args.snapshot_prefix)
        saver.save(sess, checkpoint_path, global_step= epoch)


      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      coord.request_stop()
      coord.join(threads)

def main():
  parser = argparse.ArgumentParser(description='Process Command-line Arguments')
  parser.add_argument('--load_size', nargs= 2, default= [256,256], type= int, action= 'store', help= 'The width and height of images for loading from disk')
  parser.add_argument('--crop_size', nargs= 2, default= [224,224], type= int, action= 'store', help= 'The width and height of images after random cropping')
  parser.add_argument('--batch_size', default= 128, type= int, action= 'store', help= 'The training batch size')
  parser.add_argument('--num_classes', default= 48, type= int, action= 'store', help= 'The number of classes for target problem')
  parser.add_argument('--num_channels', default= 3, type= int, action= 'store', help= 'The number of channels in input images')
  parser.add_argument('--num_epochs'  , default= 10, type= int, action= 'store', help= 'The number of epochs')
  parser.add_argument('--num_batches' , default= -1, type= int, action= 'store', help= 'The number of batches per epoch')
  parser.add_argument('--path_prefix' , default= './', action= 'store', help= 'The prefix address for images')
  parser.add_argument('--delimiter', default= ',', action = 'store', help= 'Delimiter of the labels')
  parser.add_argument('--data_info'  , default= 'tf_train.csv', action='store', help= 'File containing the addresses and labels of training images')
  parser.add_argument('--shuffle', default= True, type= bool, action= 'store', help= 'Shuffle training data or not')
  parser.add_argument('--num_threads', default= 20, type= int, action= 'store', help= 'The number of threads for loading data')
  parser.add_argument('--load_pretrained_dir', default= 'checkpoints', action= 'store', help= 'Path for loading the pretrained model')
  parser.add_argument('--log_dir', default= None, action= 'store', help= 'Path for saving Tensorboard info and checkpoints')
  parser.add_argument('--snapshot_prefix', default= 'snapshot', action= 'store', help= 'Prefix for checkpoint files')
  parser.add_argument('--architecture', default= 'resnet', help= 'The DNN architecture')
  parser.add_argument('--depth', default= 50, type= int, help= 'The depth of ResNet architecture')
  parser.add_argument('--run_name', default= 'Run'+str(time.strftime("-%d-%m-%Y_%H-%M-%S")), action= 'store', help= 'Name of the experiment')
  parser.add_argument('--log_device_placement', default= False, type= bool, help= 'Whether to log device placement or not')
  parser.add_argument('--log_debug_info', default= False, type= bool, help= 'Log the runtime information')
  parser.add_argument('--finetune', default = False, type= bool, help= 'Finetune the models or Just use it as a fixed feature extractor')
  args = parser.parse_args()

  args.chunked_batch_size = args.batch_size
  # Counting number of training examples
  args.num_samples = sum(1 for line in open(args.data_info))

  # set the number of batches per epoch
  if args.num_batches==-1:
    args.num_batches= int(args.num_samples/args.batch_size)+1

  # creating the logging directory
  if args.log_dir is None:
    args.log_dir= args.architecture+"_"+args.run_name

  print(args)
  print("Saving everything in "+args.log_dir)

  transfer(args)

if __name__ == '__main__':
  main()
