import torch.nn as nn
from .utils import get_activation

class RNN(nn.Module):
    def __init__(self, layer_specs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.rnn = None
        self.post_rnn_layers = []
        self.direction_factor = 0

        for spec in layer_specs:
            if spec["type"].lower() == "rnn":
                self.rnn = nn.RNN(
                    input_size=spec["input_size"],
                    hidden_size=spec["hidden_size"],
                    num_layers=spec.get("num_layers", 1),
                    batch_first=True,
                    bidirectional=spec.get("bidirectional", False)
                )
                self.direction_factor =2 if spec.get("bidirectional", False) else 1
            elif spec["type"].lower() == "activation":
                self.post_rnn_layers.append(get_activation(spec["name"]))
            elif spec["type"].lower() == "linear":
                if self.direction_factor != 0:
                    self.post_rnn_layers.append(nn.Linear(spec["in_features"] * self.direction_factor, spec["out_features"]))
                    self.direction_factor = 0

        self.post_rnn = nn.Sequential(*self.post_rnn_layers)

    def forward(self, x):
        x, _ = self.rnn(x)  # RNN returns (output, hidden)
        x = x[:, -1, :]     # use the output of the last time step
        return self.post_rnn(x)
