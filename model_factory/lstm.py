import torch
import torch.nn as nn
from .utils import get_activation

class LSTM(nn.Module):
    def __init__(self, layer_specs):
        super().__init__()
        layers = []
        self.direction_factor = 0
        self.is_lstm = False
        for spec in layer_specs:
            layer_type = spec["type"].lower()
            if layer_type == "lstm":
                self.is_lstm = True
                self.input_size = spec.get("input_size", 16)
                self.hidden_size = spec.get("hidden_size", 128)
                self.num_layers = spec.get("num_layers", 1)
                self.bidirectional = spec.get("bidirectional", False)
                self.batch_first = spec.get("batch_first", True),
                self.lstm = nn.LSTM(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    batch_first=self.batch_first,
                    bidirectional=self.bidirectional
                )
                self.direction_factor = 2 if self.bidirectional else 1
            elif layer_type == "linear":
                if self.direction_factor != 0:
                    layers.append(nn.Linear(spec["in_features"] * self.direction_factor, spec["out_features"]))
                    self.direction_factor = 0
            elif layer_type == "activation":
                layers.append(get_activation(spec["name"]))
            elif layer_type == "dropout":
                layers.append(nn.Dropout(spec["dropout"]))
        self.post_lstm = nn.Sequential(*layers)

    def forward(self, x):
        if self.is_lstm:
            x, _ = self.lstm(x)  # output shape: (batch, seq_len, hidden_size * directions)
            x = x[:, -1, :]  # take last time step
        return self.post_lstm(x)