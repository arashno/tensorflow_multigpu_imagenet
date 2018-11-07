# tensorflow_multigpu_imagenet
Code for training different architectures of image classification (i.e. DenseNet, ResNet, AlexNet, GoogLeNet, VGG, NiN) on ImageNet or other large datasets + Multi-GPU support + Transfer Learning support

This repository provides an easy-to-use way for training different well-known deep learning architectures on different large datasets.
The code reads dataset information from a text or csv file and directly loads images from disk. Moreover, multi-GPU and transfer learning are supported.

**************************
**New features are added**\
Completely redesigned, now more object oriented\
Now compatible with both Python 2.7 and Python 3.6\
Efficient snapshot saving\
More options for selecting optimization algorithm\
More options for learning rate and weight decay policies\
Tuned architectures and bug fix\
More readable code
Now supports named classes in CSV file
**************************

This code got inspiration from these repositories:

https://github.com/soumith/imagenet-multiGPU.torch

https://github.com/ry/tensorflow-resnet

https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10



#Example of usages:

To start, an input text file is needed. In the input text file, each line contain an image address and its associated label in numeric form:

train/n01440764/n01440764_7173.JPEG,0\
train/n01440764/n01440764_3724.JPEG,0\
train/n01440764/n01440764_7719.JPEG,0\
train/n01440764/n01440764_7304.JPEG,0\
train/n01440764/n01440764_8469.JPEG,0

Use the --delimiter option to specify the delimiter character, and --path_prefix to add a constant prefix to all the paths.

For training execute run.py with train command and your appropriate argument. For example, to train the VGG architecture on the ImageNet dataset with Adam optimizer for 50 epochs execute this: 

```bash
python run.py train --architecture vgg --path_prefix /path..to..train/ --train_info train.txt --optimizer adam --num_epochs 50 
```

To evaluate a trained model (example):

```bash
python run.py eval --num_threads 8 --architecture alexnet --log_dir "alexnet_Run-17-07-2017-15:31:57" --path_prefix /project/datasets/imagenet/train/ --val_info val.txt
```

To test a trained model on data (example):

```bash
python run.py inference --num_threads 8 --architecture alexnet --log_dir "alexnet_Run-17-07-2017-15:31:57" --path_prefix /project/datasets/imagenet/train/ --val_info val.txt --save_predictions preds.txt
```

Transfer learning (example):

```bash
python run.py train --transfer_mode 1 --architecture alexnet --retrain_from ./alexnet_Run-17-07-2017-15:31:57 --optimizer momentum --LR_policy constant --LR_details 0.001
```
