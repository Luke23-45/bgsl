"""
models/__init__.py
"""
from bgsl.models.gru import GRUPredictor
from bgsl.models.tcn import TCNPredictor
from bgsl.models.transformer import TransformerPredictor

__all__ = ["GRUPredictor", "TCNPredictor", "TransformerPredictor"]

REGISTRY = {
    "gru": GRUPredictor,
    "tcn": TCNPredictor,
    "transformer": TransformerPredictor,
}


def build_model(name: str, **kwargs):
    """
    Build a model by name.

    Parameters
    ----------
    name : str
        One of "gru", "tcn", "transformer".
    **kwargs
        Passed to the model constructor.
    """
    name = name.lower()
    if name not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
