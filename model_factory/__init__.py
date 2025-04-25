from .mlp import MLP
from .cnn import CNN
from .rnn import RNN
from .lstm import LSTM
from .transformer import Transformer
from .utils import get_activation, validate_layer_specs

MODEL_BUILDERS = {
    "mlp": MLP,
    "cnn": CNN,
    "rnn": RNN,
    "lstm": LSTM,
    "transformer": Transformer,
}

def build_model(model_type, layer_specs=None):
    model_type = model_type.lower()
    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported model type: {model_type}")

    validate_layer_specs(model_type, layer_specs)
    print("Validated layer spec successfully ✅")
    return MODEL_BUILDERS[model_type](layer_specs)