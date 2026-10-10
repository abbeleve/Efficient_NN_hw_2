import torch.nn as nn

from . import resnet_cifar10
from .lenet import LeNet

MODELS = {
    "resnet20": resnet_cifar10.resnet20,
    "resnet32": resnet_cifar10.resnet32,
    "resnet44": resnet_cifar10.resnet44,
    "resnet56": resnet_cifar10.resnet56,
    "resnet110": resnet_cifar10.resnet110,
    "lenet": LeNet,
}


def get_model(name: str, num_classes: int, bn_momentum: float | None = None) -> nn.Module:
    """Builds a model; optionally overrides the BatchNorm running-stats momentum.

    PyTorch's BN `momentum` is the weight of the *new* batch statistic, i.e.
    1 - moving_average_fraction in Caffe terms: the paper's MAF=0.95 is
    momentum=0.05, Caffe's default 0.999 is momentum=0.001.
    """
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}, expected one of {sorted(MODELS)}")
    model = MODELS[name](num_classes=num_classes)
    if bn_momentum is not None:
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.momentum = bn_momentum
    return model
