# tensorflow_multigpu_imagenet
Code for training different architectures( DenseNet, ResNet, AlexNet, GoogLeNet, VGG, NiN) on ImageNet dataset + Multi-GPU support + Transfer Learning support

This repository provides an easy-to-use way for training different well-known deep learning architectures on different datasets.
The code directly load images from disk. Moreover, multi-GPU and transfer learning is also supported.
This code is mainly based on these repositories:

https://github.com/soumith/imagenet-multiGPU.torch

https://github.com/ry/tensorflow-resnet

https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10



#Example of usages:

Training:

python train.py --path_prefix /project/datasets/imagenet/train/

Evaluating a trained model:

python eval.py --num_threads 8 --architecture alexnet --log_dir "alexnet_Run-17-07-2017-15:31:57" --path_prefix /project/datasets/imagenet/train/

Transfer learning:

python transfer.py --architecture alexnet --load_pretrained_dir ./alexnet_Run-17-07-2017-15:31:57

Evaluate a transferred model:

python transfer_eval.py --num_threads 4 --architecture alexnet  --log_dir ./alexnet_Run-18-07-2017-14:08:14 --delimiter , --save_predictions trnpred.txt --path_prefix /project/dataset2
