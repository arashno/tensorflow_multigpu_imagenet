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

def evaluate(args):

  with tf.Graph().as_default() as g, tf.device('/cpu:0'):
    # Get images and labels for CIFAR-10.
    if args.save_predictions is None:
      images, labels = data_loader.read_inputs(False, args)
    else:
      images, labels, urls = data_loader.read_inputs(False, args)

    with tf.device('/gpu:0'):
       # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = arch.get_model(images, 0.0, False, args, transferMode=True)

        top_1_op = tf.nn.in_top_k(logits, labels, 1)
        top_5_op = tf.nn.in_top_k(logits, labels, 5)

        if args.save_predictions is not None:
          top5 = tf.nn.top_k(tf.nn.softmax(logits), 5)
          top5ind= top5.indices
          top5val= top5.values
 
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(args.log_dir, g)

    with tf.Session(config=tf.ConfigProto( allow_soft_placement= True)) as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())

      ckpt = tf.train.get_checkpoint_state(args.log_dir)

      # Load the latest model
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)

      else:
        return
   
      # Start the queue runners.
      coord = tf.train.Coordinator()

      threads = tf.train.start_queue_runners(sess= sess, coord= coord)
      true_count = 0  # Counts the number of correct predictions
      true_count5 = 0
      all_count = 0
      step = 0
      predictions_format_str = ('%d,%s,%d,%s,%s\n')
      batch_format_str = ('Batch Number: %d, Top-1 Hit: %d, Top-5 Hit: %d, Top-1 Accuracy: %.3f, Top-5 Accuracy: %.3f')

      if args.save_predictions is not None:
        out_file = open(args.save_predictions, 'w')
 
      while step < args.num_batches and not coord.should_stop():
        if args.save_predictions is None:
          predictions, predictions5 = sess.run([top_1_op, top_5_op])
        else:
          predictions, predictions5, urls_values, label_values, top5preds, top5conf = sess.run([top_1_op, top_5_op, urls, labels, top5ind, top5val])
          for i in xrange(0, urls_values.shape[0]):
            out_file.write(predictions_format_str%(step*args.batch_size+i+1, urls_values[i], label_values[i],
                '[' + ', '.join('%d' % item for item in top5preds[i]) + ']',
                '[' + ', '.join('%.4f' % item for item in top5conf[i]) + ']'))
            out_file.flush()
        true_count += np.sum(predictions)
        true_count5 += np.sum(predictions5)
        all_count += predictions.shape[0]
        print(batch_format_str%(step, true_count, true_count5, true_count / all_count, true_count5 / all_count))
        sys.stdout.flush()
        step += 1

      if args.save_predictions is not None:
        out_file.close()
 
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      coord.request_stop()
      coord.join(threads)

def main():
  parser = argparse.ArgumentParser(description='Process Command-line Arguments')
  parser.add_argument('--load_size', nargs= 2, default= [256,256], type= int, action= 'store', help= 'The width and height of images for loading from disk')
  parser.add_argument('--crop_size', nargs= 2, default= [224,224], type= int, action= 'store', help= 'The width and height of images after random cropping')
  parser.add_argument('--batch_size', default= 512, type= int, action= 'store', help= 'The testing batch size')
  parser.add_argument('--num_classes', default= 48, type= int, action= 'store', help= 'The number of classes')
  parser.add_argument('--num_channels', default= 3, type= int, action= 'store', help= 'The number of channels in input images')
  parser.add_argument('--num_batches' , default= -1, type= int, action= 'store', help= 'The number of batches of data')
  parser.add_argument('--path_prefix' , default= './', action='store', help= 'The prefix address for images')
  parser.add_argument('--delimiter' , default= ',', action = 'store', help= 'Delimiter for the input files')
  parser.add_argument('--data_info'  , default= 'gold_expert_info.csv', action='store', help= 'File containing the addresses and labels of testing images')
  parser.add_argument('--num_threads', default= 8, type= int, action= 'store', help= 'The number of threads for loading data')
  parser.add_argument('--log_dir', default= None, action= 'store', help= 'Path for saving Tensorboard info and checkpoints')
  parser.add_argument('--architecture', default= 'resnet', help= 'The DNN architecture')
  parser.add_argument('--depth', default= 50, type= int, help= 'The depth of ResNet architecture')
  parser.add_argument('--run_name', default= 'Run'+str(time.strftime("-%d-%m-%Y-%H:%M:%S")), action= 'store', help= 'Name of the experiment')
  parser.add_argument('--log_device_placement', default= False, type= bool, help= 'Whether to log device placement or not')
  parser.add_argument('--save_predictions', default= None, action= 'store', help= 'Save top-5 predictions of the networks along with their confidence in the specified file')
  args = parser.parse_args()
  args.chunked_batch_size = args.batch_size
  args.num_samples = sum(1 for line in open(args.data_info))
  if args.num_batches==-1:
    if(args.num_samples%args.batch_size==0):
      args.num_batches= int(args.num_samples/args.batch_size)
    else:
      args.num_batches= int(args.num_samples/args.batch_size)+1

  print(args)

  evaluate(args)


if __name__ == '__main__':
  main()
