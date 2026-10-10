"""LeNet as shipped with Caffe (examples/mnist/lenet_train_test.prototxt).

The paper's MNIST experiments (Table 2) use this network with Caffe's default
hyper-parameters. Note that the Caffe version has no non-linearity after the
convolutions; the only ReLU sits between the two fully connected layers.
"""

import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),   # 28 -> 24
            nn.MaxPool2d(2),                   # 24 -> 12
            nn.Conv2d(20, 50, kernel_size=5),  # 12 -> 8
            nn.MaxPool2d(2),                   # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
