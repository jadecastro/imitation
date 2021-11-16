"""Helper methods to build and run neural networks."""

import collections
from typing import Iterable, Optional, Type

from torch import nn


class SqueezeLayer(nn.Module):
    """Torch module that squeezes a B*1 tensor down into a size-B vector."""

    def forward(self, x):
        assert x.ndim == 2 and x.shape[1] == 1
        new_value = x.squeeze(1)
        assert new_value.ndim == 1
        return new_value


def build_mlp(
    in_size: int,
    hid_sizes: Iterable[int],
    out_size: int = 1,
    name: Optional[str] = None,
    activation: Type[nn.Module] = nn.ReLU,
    squeeze_output=False,
    flatten_input=False,
) -> nn.Module:
    """Constructs a Torch MLP.

    Args:
        in_size: size of individual input vectors; input to the MLP will be of
            shape (batch_size, in_size).
        hid_sizes: sizes of hidden layers. If this is an empty iterable, then we build
            a linear function approximator.
        out_size: required size of output vector.
        name: Name to use as a prefix for the layers ID.
        activation: activation to apply after hidden layers.
        squeeze_output: if out_size=1, then squeeze_input=True ensures that MLP
            output is of size (B,) instead of (B,1).
        flatten_input: should input be flattened along axes 1, 2, 3, …? Useful
            if you want to, e.g., process small images inputs with an MLP.

    Returns:
        nn.Module: an MLP mapping from inputs of size (batch_size, in_size) to
            (batch_size, out_size), unless out_size=1 and squeeze_output=True,
            in which case the output is of size (batch_size, ).

    Raises:
        ValueError: if squeeze_output was supplied with out_size!=1.
    """
    layers = collections.OrderedDict()

    if name is None:
        prefix = ""
    else:
        prefix = f"{name}_"

    if flatten_input:
        layers[f"{prefix}flatten"] = nn.Flatten()

    # Hidden layers
    prev_size = in_size
    for i, size in enumerate(hid_sizes):
        layers[f"{prefix}dense{i}"] = nn.Linear(prev_size, size)
        prev_size = size
        if activation:
            layers[f"{prefix}act{i}"] = activation()

    # Final layer
    layers[f"{prefix}dense_final"] = nn.Linear(prev_size, out_size)

    if squeeze_output:
        if out_size != 1:
            raise ValueError("squeeze_output is only applicable when out_size=1")
        layers[f"{prefix}squeeze"] = SqueezeLayer()

    model = nn.Sequential(layers)

    return model


def build_gaussian_lstm(
    in_size: int,
    hidden_size: int,
    latent_size: int,
    num_layers: int = 1,
    bidirectional: bool = False,
    name: Optional[str] = None,
    squeeze_output=False,
    flatten_input=False,
) -> nn.Module:
    """Constructs a Torch LSTM.

    Args:
        in_size: size of individual input vectors; input to the LSTM will be of
            shape (batch_size, in_size).
        hidden_size: size of hidden layers.
        latent_size: size of the latent space.
        num_layers: number of hidden layers.
        bidirectional: if True, then configures the LSTM as bidirectional.
        name: Name to use as a prefix for the layers ID.
        flatten_input: should input be flattened along axes 1, 2, 3, …? Useful
            if you want to, e.g., process small images inputs.

    Returns:
        nn.Module: an LSTM mapping from inputs of size (batch_size, in_size) to
            (batch_size, out_size), unless out_size=1 and squeeze_output=True,
            in which case the output is of size (batch_size, ).

    Raises:
        ValueError: if squeeze_output was supplied with out_size!=1.
    """
    hidden_factor = (2 if bidirectional else 1) * num_layers

    layers = collections.OrderedDict()

    if name is None:
        prefix = ""
    else:
        prefix = f"{name}_"

    if flatten_input:
        layers[f"{prefix}flatten"] = nn.Flatten()

    layers[f"{prefix}lstm"] = nn.LSTM(
        input_size=in_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        batch_first=True,
    )

    model = nn.Sequential(layers)

    hidden_to_mean = nn.Linear(hidden_size * hidden_factor, latent_size)
    hidden_to_logstd = nn.Linear(hidden_size * hidden_factor, latent_size)

    # TODO(jon): Sequence these into a single network rather than spit out 3 objects.
    return model, hidden_to_mean, hidden_to_logstd
