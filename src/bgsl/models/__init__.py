"""
models/__init__.py
------------------
Pure PyTorch neural network backbones for BGSL risk predictors.
All models share the same forward signature:

    forward(x, mask=None, seq_lens=None) -> logits [B, T]

Backbone selection is handled entirely by LightningCLI via
`class_path` in experiment YAML configs — no factory needed here.
"""

from bgsl.models.gru import GRUPredictor
from bgsl.models.tcn import TCNPredictor
from bgsl.models.transformer import TransformerPredictor
from bgsl.models.hypernetwork import HyperNetwork

__all__ = ["GRUPredictor", "TCNPredictor", "TransformerPredictor", "HyperNetwork"]
