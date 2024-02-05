#!/usr/bin/env python
"""This script ...
"""
import logging
from typing import Tuple

import click

from deep_tensors.xor import xor_classifier

_log = logging.getLogger(__name__)
Units = Tuple[int, ...]
ACTIVATION = [
    "relu",
    "sigmoid",
    "softmax",
    "tanh",
]
Activations = Tuple[xor_classifier.nn.Activation, ...]


@click.command()
@click.option(
    "--units",
    "-u",
    type=click.INT,
    multiple=True,
    default=(1, ),
    show_default=True,
    help="Count of units in the layer of the net",
)
@click.option(
    "--activation",
    "-a",
    type=click.Choice(ACTIVATION),
    multiple=True,
    default=("relu", ),
    show_choices=True,
    show_default=True,
    help="Choose the activation function",
)
def cli(units: Units, activation: Activations):
    """Client"""
    if len(units) != len(activation):
        raise ValueError("Units and activations must have the same length")
    print(units)
    print(activation)
    xor = xor_classifier.XOR(
        units=units,
        activations=activation,
        train_count=100,
        epochs=500,
        batch_size=32,
    )
    xor.train()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
