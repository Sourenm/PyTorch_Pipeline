import torch
import torch.nn as nn
from .utils import get_activation

class MLP(nn.Module):
    def __init__(self, layer_specs):
        super().__init__()
        layers = []
        for spec in layer_specs:
            layer_type = spec["type"].lower()
            if layer_type == "linear":
                layers.append(nn.Linear(spec["in_features"], spec["out_features"]))
            elif layer_type == "activation":
                layers.append(get_activation(spec["name"]))
            elif layer_type == "dropout":
                layers.append(nn.Dropout(p=spec["dropout"]))
            elif layer_type == "batchnorm":
                layers.append(nn.BatchNorm1d(spec["out_features"]))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
