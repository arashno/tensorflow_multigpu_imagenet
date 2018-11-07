from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf


"""
 This class models an input pipeline which takes the input from a .csv file and return batches of images and their labels.
 For more details, please read inline comments below.
"""
class loader:

  """
   Constructor. it gathers the necessary information to build a data loader pipeline.
   For more details, please read inline comments below.

   Args:
     input_file: a file contains input images paths and their labels (one in a row).
     delimiter: the delimiter character separating between file paths and their labels
     raw_size: the images read from disk will be resized to this size before further processing
     processed_size: the size of input images after preprocessing
     is_training: determine appying either the training preprocessing steps or testing preprocessing steps.
     batch_size: batch size
     num_prefetch: number of examples to load in memory
     num_thtreads: number of loader threads
     path-prefix: a common prefix to path of all the input images
     shuffle: shuffle the training images or not
     inference_only: if True, no label is provided and just read imge paths from input file
   Returns:
     nothing.
  """
  def __init__(self, input_file, delimiter, raw_size, processed_size, is_training, batch_size, num_prefetch, num_threads, path_prefix, shuffle=False, inference_only= False):
    self.input_file= input_file
    self.delimiter= delimiter
    self.raw_size= raw_size
    self.processed_size= processed_size
    self.is_training= is_training
    self.batch_size= batch_size
    self.num_prefetch= num_prefetch
    self.num_threads= num_threads
    self.shuffle= shuffle
    self.path_prefix= path_prefix
    self.inference_only= inference_only

  """
   This method reads and parses the input file and return two lists of imge paths and their labels

   Args:
     nothing
   Returns:
     two lists for imag paths and their labels.
  """
  def _read_label_file(self):
    f = open(self.input_file, "r")
    filepaths = []
    # if no label is provided just read input file names
    if not self.inference_only:
      labels = []
      # loop to read the rows
      for line in f:
        tokens = line.split(self.delimiter)
        filepaths.append(tokens[0])
        labels.append(tokens[1].rstrip().lower())
      # assign class ids
      self.label_set = set(labels)
      if all([ x.isdigit() for x in self.label_set]):
         print("found %d classes"%(len(self.label_set)))
         self.label_dict= {int(x): int(x) for x in sorted(self.label_set)}
         labels= [int(x) for x in labels]
      else:
         print("found %d classes"%(len(self.label_set)))
         self.label_dict= {i: x for i,x in enumerate(sorted(self.label_set))}
         labels= [dict(zip(self.label_dict.values(), self.label_dict.keys()))[x] for x in labels]
      print(self.label_dict)
      # return results
      return filepaths, labels

    else:
      # loop to read rows
      for line in f:
          filepaths.append(line.rstrip())
      # return results
      return filepaths, None

  """
   This method takes a filename, read the image, do preprocessing, and return the result
   For more details, please read inline comments below.

   Args:
     filename : path of an input image
   Returns:
     preprocessed image
  """
  def preprocess(self, filename):
    # Read examples from files in the filename queue.
    file_content = tf.read_file(filename)
    # Read JPEG or PNG or GIF image from file
    reshaped_image = tf.to_float(tf.image.decode_jpeg(file_content, channels=self.raw_size[2]))
    # Resize image to 256*256
    reshaped_image = tf.image.resize_images(reshaped_image, (self.raw_size[0], self.raw_size[1]))

    img_info = filename

    if self.is_training:
      reshaped_image = self._train_preprocess(reshaped_image)
    else:
      reshaped_image = self._test_preprocess(reshaped_image)

    # Subtract off the mean and divide by the variance of the pixels.
    reshaped_image = tf.image.per_image_standardization(reshaped_image)

    # Set the shapes of tensors.
    reshaped_image.set_shape(self.processed_size)

    return reshaped_image

  """
  This method makes the input pipeline for reading data.
  For more details, please check the inline comments.

  Args:
    nothing.
  Returns:
    image and label batches
  """
  def load(self):
    # read and parse the input file
    filepaths, labels = self._read_label_file()

    # add path prefix to all image paths
    filenames = [os.path.join(self.path_prefix,i) for i in filepaths]
    
    # Create a queue that produces the filenames to read.
    if not self.inference_only:
      # amke FIFO queue of file paths and their labels
      filename_queue = tf.train.slice_input_producer([filenames, labels], shuffle= self.shuffle if self.is_training else False)

      image_queue= filename_queue[0]
      label_queue= filename_queue[1]

      # prprocess images
      reshaped_image = self.preprocess(image_queue)

      # label
      label = tf.cast(label_queue, tf.int64)
      img_info = image_queue

      print ('Filling queue with %d images before starting to train. '
         'This may take some times.' % self.num_prefetch)

      # Load images and labels with additional info and return batches
      return tf.train.batch(
        [reshaped_image, label, img_info],
        batch_size= self.batch_size,
        num_threads= self.num_threads,
        capacity=self.num_prefetch,
        allow_smaller_final_batch=True if not self.is_training else False)

    else:

      filename_queue = tf.train.slice_input_producer([filenames], shuffle= self.shuffle if self.is_training else False)
      image_queue= filename_queue[0]
      reshaped_image = self.preprocess(image_queue)
      img_info = image_queue

      print ('Filling queue with %d images before starting to train. '
         'This may take some times.' % self.num_prefetch)

      # Load images and labels with additional info and return batches
      return tf.train.batch(
        [reshaped_image, img_info],
        batch_size= self.batch_size,
        num_threads= self.num_threads,
        capacity=self.num_prefetch,
        allow_smaller_final_batch=True if not self.is_training else False)

  """
   This method applies different data aufmentation techniques to an input image
   For more details, please read inline comments below.

   Args:
     reshaped_image: input image
   Returns:
     augmentaed image
  """
  def _train_preprocess(self, reshaped_image):
    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    reshaped_image = tf.random_crop(reshaped_image, self.processed_size)

    # Randomly flip the image horizontally.
    reshaped_image = tf.image.random_flip_left_right(reshaped_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    reshaped_image = tf.image.random_brightness(reshaped_image,
                                               max_delta=63)
    # Randomly changing contrast of the image
    reshaped_image = tf.image.random_contrast(reshaped_image,
                                             lower=0.2, upper=1.8)
    return reshaped_image

  """
   This method centrally crops an input image using the the provided config

   Args:
     reshaped_image: input image
   Returns:
     centrally cropped image
  """
  def _test_preprocess(self, reshaped_image):

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         self.processed_size[0], self.processed_size[1])

    return resized_image
