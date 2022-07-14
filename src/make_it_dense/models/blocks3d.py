from typing import List

import torch.nn as nn


class ResNetBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out


class Unet3D(nn.Module):
    def __init__(self, channels: List[int], layers_down: List[int], layers_up: List[int]):
        super().__init__()

        self.layers_down = nn.ModuleList()
        self.layers_down.append(ResNetBlock3d(channels[0], channels[0]))
        for i in range(1, len(channels)):
            layer = [
                nn.Conv3d(
                    channels[i - 1], channels[i], kernel_size=3, stride=2, padding=1, bias=False
                ),
                nn.BatchNorm3d(channels[i]),
                nn.ReLU(inplace=True),
            ]
            # Do we need 4 resnet blocks here?
            layer += [ResNetBlock3d(channels[i], channels[i]) for _ in range(layers_down[i])]
            self.layers_down.append(nn.Sequential(*layer))

        channels = channels[::-1]
        self.layers_up_conv = nn.ModuleList()
        for i in range(1, len(channels)):
            self.layers_up_conv.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        channels[i - 1], channels[i], kernel_size=2, stride=2, bias=False
                    ),
                    nn.BatchNorm3d(channels[i]),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(channels[i], channels[i], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm3d(channels[i]),
                    nn.ReLU(inplace=True),
                )
            )

        self.layers_up_res = nn.ModuleList()
        for i in range(1, len(channels)):
            layer = [ResNetBlock3d(channels[i], channels[i]) for _ in range(layers_up[i - 1])]
            self.layers_up_res.append(nn.Sequential(*layer))

    def forward(self, x):
        xs = []
        for layer in self.layers_down:
            x = layer(x)
            xs.append(x)

        xs.reverse()
        out = []
        for i in range(len(self.layers_up_conv)):
            x = self.layers_up_conv[i](x)
            x = (x + xs[i + 1]) / 2.0
            x = self.layers_up_res[i](x)
            out.append(x)

        return out


class FeatureExtractor(nn.Module):
    """Extract features from a TSDF volume withouth chaning the size."""

    def __init__(self, channels=4):
        super().__init__()
        self.model = ResNetBlock3d(1, channels)

    def forward(self, x):
        return self.model(x)
