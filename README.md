# tensorflow_multigpu_imagenet
Code for training different architectures( DenseNet, ResNet, AlexNet, GoogLeNet, VGG, NiN) on ImageNet or other datasets + Multi-GPU support + Transfer Learning support

This repository provides an easy-to-use way for training different well-known deep learning architectures on different datasets.
The code reads dataset information from a text or csv file and directly loads images from disk. Moreover, multi-GPU and transfer learning are supported.

This code takes advantage of these repositories:

https://github.com/soumith/imagenet-multiGPU.torch

https://github.com/ry/tensorflow-resnet

https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10



#Example of usages:

Training:

python train.py --architecture alexnet --path_prefix /project/datasets/imagenet/train/ --data_info train.txt

Evaluating a trained model:

python eval.py --num_threads 8 --architecture alexnet --log_dir "alexnet_Run-17-07-2017-15:31:57" --path_prefix /project/datasets/imagenet/train/ --data_info val.txt

Transfer learning:

python train.py --transfer_mode 1 --architecture alexnet --retrain_from ./alexnet_Run-17-07-2017-15:31:57
