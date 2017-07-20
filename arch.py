import architectures.alexnet
import architectures.resnet

def get_model(inputs, wd, is_training, args, transferMode= False):
    if args.architecture=='alexnet':
        return architectures.alexnet.inference(inputs, args.num_classes, wd, 0.5 if is_training else 1.0, is_training, transferMode)
    elif args.architecture=='resnet':
        return architectures.resnet.inference(inputs, args.resnet_depth, args.num_classes, wd, is_training, transferMode)
