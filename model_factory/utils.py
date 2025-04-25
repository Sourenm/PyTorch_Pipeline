import torch.nn as nn
import math

def get_activation(name):
    """
    Returns a PyTorch activation function based on a given name.

    Args:
        name (str): The name of the activation function. Supported values:
            'relu', 'tanh', 'sigmoid', 'leaky_relu', 'gelu'.

    Returns:
        torch.nn.Module: Corresponding activation function. Defaults to ReLU if the name is unrecognized.
    """    
    return {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(0.1),
        "gelu": nn.GELU(),
    }.get(name.lower(), nn.ReLU())

def init_weights(m, method="kaiming", nonlinearity="relu"):
    """
    Initializes weights and biases properly for a given Linear layer.

    Args:
        m (nn.Module): Module to initialize.
        method (str): 'kaiming' or 'xavier'.
        nonlinearity (str): Activation following this layer (e.g., 'relu').
    """
    if isinstance(m, nn.Linear):
        if method == "kaiming":
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
        elif method == "xavier":
            nn.init.xavier_uniform_(m.weight)
        else:
            raise ValueError("Unsupported init method")

        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

def validate_layer_specs(model_type, layer_spec):
    """
    Validates the given layer specification based on model type.
    Raises ValueError with human-readable errors for invalid formats.
    """
    if not isinstance(layer_spec, list):
        raise ValueError("Layer specification must be a list of layer dictionaries.")

    for i, layer in enumerate(layer_spec):
        if not isinstance(layer, dict):
            raise ValueError(f"Layer at index {i} is not a dictionary.")        
        if model_type == "cnn":
            if "type" not in layer:
                raise ValueError(f"Missing 'type' key in CNN layer at index {i}")
            if layer["type"].lower() not in ["conv", "maxpool", "flatten", "linear", "dropout", "conv2d", "maxpool2d", "activation"]:                
                raise ValueError(f"Invalid CNN layer type: {layer['type']} at index {i}")
            if layer["type"].lower() == "activation":
                if layer["name"].lower() not in ["relu", "sigmoid", "tanh", "softmax", "leaky_relu"]:
                    raise ValueError(f"Invalid activation layer type: {layer['type']['name']}, allowed types are ReLU, Sigmoid, Tanh, Softmax, Leaky_ReLU")

        elif model_type in ["rnn", "lstm"]:
            if "type" not in layer:
                raise ValueError(f"Missing 'type' key in RNN, LSTM layer at index {i}")            
            if layer["type"].lower() == "rnn" or layer["type"].lower() == "lstm":
                if "hidden_size" not in layer or "num_layers" not in layer:
                    raise ValueError(f"RNN/LSTM layer at index {i} must have 'hidden_size' and 'num_layers'")

        elif model_type == "transformer":            
            if "type" not in layer:
                raise ValueError(f"Missing 'type' key in RNN, LSTM layer at index {i}")            
            if layer["type"].lower() == "transformer_encoder":
                required_keys = {"type", "d_model", "num_heads", "num_layers"}
                missing = required_keys - layer.keys()
                if missing:
                    raise ValueError(f"Transformer layer at index {i} is missing keys: {', '.join(missing)}")
            if layer["type"].lower() == "embedding":
                required_keys = {"type", "in_features", "out_features"}
                missing = required_keys - layer.keys()
                if missing:
                    raise ValueError(f"Transformer layer at index {i} is missing keys: {', '.join(missing)}")        

        elif model_type == "mlp":
            if "type" not in layer:
                raise ValueError(f"Missing 'type' key in MLP layer at index {i}")                        
            if layer["type"].lower() == "mlp":
                if "out_features" not in layer:
                    raise ValueError(f"MLP layer at index {i} must include 'out_features'")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

def config_to_code(config):
    """
    Converts model configuration to a string of Python code for logging/reproducibility.
    """
    from pprint import pformat

    return f"""# Reproducible config
model_type = "{config["model_type"]}"
input_dim = {config.get("input_shape", None)}
output_dim = {config.get("output_shape", None)}
layer_spec = {pformat(config["layer_specs"])}
"""