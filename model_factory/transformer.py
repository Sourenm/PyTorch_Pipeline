import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, layer_specs):
        super().__init__()
        self.model = nn.Sequential()
        self.embedding = None
        for spec in layer_specs:
            layer_type = spec["type"].lower()
            if layer_type == "embedding":
                self.embedding = nn.Embedding(spec["num_embeddings"], spec["embedding_dim"])
            elif layer_type == "transformer_encoder":
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=spec["d_model"],
                    nhead=spec["num_heads"],
                    dim_feedforward=spec.get("dim_feedforward", 2048),
                    dropout=spec.get("dropout", 0.1),
                    activation=spec.get("activation", "relu")
                )
                self.model = nn.TransformerEncoder(encoder_layer, num_layers=spec["num_layers"])
            elif layer_type == "linear":
                self.output_layer = nn.Linear(spec["in_features"], spec["out_features"])

    def forward(self, x):
        if self.embedding:
            x = self.embedding(x)
        x = self.model(x)
        x = x.mean(dim=1)  # global average pooling
        return self.output_layer(x)