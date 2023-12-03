import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DeconvMobileNet(nn.Module):
    def __init__(self, num_classes, init_weights):
        super(DeconvMobileNet, self).__init__()

        mobilenet = models.mobilenet_v3_small(pretrained=True)
        features = list(mobilenet.features.children())
        classifier = list(mobilenet.classifier.children())

        # Extracting layers from MobileNetV3
        self.conv1 = nn.Sequential(features[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, return_indices=True)# stride 1 instead of 2

        self.conv2 = nn.Sequential(features[1], features[2], features[3])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, return_indices=True)# stride 1 instead of 2

        self.conv3 = nn.Sequential(features[4], features[5], features[6])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, return_indices=True)# stride 1 instead of 2

        self.conv4 = nn.Sequential(features[7], features[8], features[9])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, return_indices=True) # stride 1 instead of 2

        self.conv5 = nn.Sequential(features[10], features[11], features[12])
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, return_indices=True) # stride 1 instead of 2

        # Conv67 layer
        self.conv67 = nn.Sequential(
            nn.Conv2d(576, 1024, kernel_size=(1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1000, kernel_size=(1, 1)),
            nn.BatchNorm2d(1000),
            nn.ReLU()
        )

        # Load weights for conv6 and conv7
        w_conv6 = classifier[0].state_dict()
        w_conv7 = classifier[3].state_dict()

        new_input_channels_conv6 = 96
        w_conv6_adjusted = w_conv6['weight'].unsqueeze(2).unsqueeze(3)  # .view(1024, 576, 1, 1)
        self.conv67[0].weight.data.copy_(w_conv6_adjusted)
        self.conv67[0].bias.data.copy_(w_conv6['bias'])
        self.conv67[3].weight.data.copy_(w_conv7['weight'].view(1000, 1024, 1, 1))
        self.conv67[3].bias.data.copy_(w_conv7['bias'])

        # Define deconvolution layers
        self.deconv67 = nn.Sequential(
            nn.ConvTranspose2d(1000, 1024, kernel_size=7, stride=1, padding=0),  # Mirrors conv67
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(1024, 576, kernel_size=3, stride=1, padding=1),  # Mirrors conv5
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(576, 576, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(576, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),  # Mirrors conv4
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),  # Mirrors conv3
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),  # Mirrors conv2
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),  # Mirrors conv1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        original = x

        x = self.conv1(x)
        x, p1 = self.pool1(x)

        x = self.conv2(x)
        x, p2 = self.pool2(x)

        x = self.conv3(x)
        x, p3 = self.pool3(x)

        x = self.conv4(x)
        x, p4 = self.pool4(x)

        x = self.conv5(x)
        x, p5 = self.pool5(x)

        x = self.conv67(x)
        x = self.deconv67(x)

        x = self.unpool5(x, p5)
        x = self.deconv5(x)

        x = self.unpool4(x, p4)
        x = self.deconv4(x)

        x = self.unpool3(x, p3)
        x = self.deconv3(x)

        x = self.unpool2(x, p2)
        x = self.deconv2(x)

        x = self.unpool1(x, p1)
        x = self.deconv1(x)

        return x

    def _initialize_weights(self):
        targets = [self.conv67, self.deconv67, self.deconv5, self.deconv4, self.deconv3, self.deconv2, self.deconv1]
        for layer in targets:
            for module in layer.children():
                if isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)


if __name__ == '__main__':
    # Instantiate the model
    num_classes = 10  # You can change this based on your actual number of classes
    model = DeconvMobileNet(num_classes=num_classes, init_weights=True)

    # Generate random input data (batch size = 1, channels = 3, height = 224, width = 224)
    input_data = torch.randn((1, 3, 224, 224))

    # Forward pass
    output = model(input_data)

    # Print the output shape
    print("Output Shape:", output.shape)
