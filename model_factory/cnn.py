import torch.nn as nn
from .utils import get_activation

class CNN(nn.Module):
    def __init__(self, layer_specs):
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = None
        for spec in layer_specs:
            layer_type = spec["type"].lower()
            if layer_type == "conv2d":
                in_channels = spec.get("in_channels", in_channels)
                out_channels = spec["out_channels"]
                self.layers.append(nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=spec.get("kernel_size", 3),
                    stride=spec.get("stride", 1),
                    padding=spec.get("padding", 1)
                ))
                in_channels = out_channels
            elif layer_type == "maxpool2d":
                self.layers.append(nn.MaxPool2d(kernel_size=spec.get("kernel_size", 2)))
            elif layer_type == "activation":
                self.layers.append(get_activation(spec["name"]))
            elif layer_type == "batchnorm":
                self.layers.append(nn.BatchNorm2d(in_channels))
            elif layer_type == "dropout":
                self.layers.append(nn.Dropout2d(spec["dropout"]))
            elif layer_type == "flatten":
                self.layers.append(nn.Flatten())
            elif layer_type == "linear":
                self.layers.append(nn.Linear(spec["in_features"], spec["out_features"]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
